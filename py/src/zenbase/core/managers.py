import inspect
from abc import ABC
from contextlib import contextmanager
from typing import Any, Callable, Union

from zenbase.types import LMFunction, LMZenbase
from zenbase.utils import ksuid


class BaseTracer(ABC):
    pass


class ZenbaseTracer(BaseTracer):
    def __init__(self):
        self.all_traces = {}
        self.current_trace = None
        self.current_key = None
        self.optimized_args = {}

    def __call__(self, function: Callable[[Any], Any] = None, zenbase: LMZenbase = None) -> Union[Callable, LMFunction]:
        if function is None:
            return lambda f: self.trace_function(f, zenbase)
        return self.trace_function(function, zenbase)

    def trace_function(self, function: Callable[[Any], Any] = None, zenbase: LMZenbase = None) -> LMFunction:
        def wrapper(request, lm_function, *args, **kwargs):
            func_name = function.__name__
            run_timestamp = ksuid(func_name)

            if self.current_trace is None:
                with self.trace_context(func_name, run_timestamp):
                    return self._execute_and_trace(function, func_name, request, lm_function, *args, **kwargs)
            else:
                return self._execute_and_trace(function, func_name, request, lm_function, *args, **kwargs)

        return LMFunction(wrapper, zenbase)

    @contextmanager
    def trace_context(self, func_name, run_timestamp, optimized_args=None):
        if self.current_trace is None:
            self.current_trace = {}
            self.current_key = run_timestamp
        if optimized_args:
            self.optimized_args = optimized_args
        try:
            yield
        finally:
            if self.current_key == run_timestamp:
                if run_timestamp not in self.all_traces:
                    self.all_traces[run_timestamp] = {}
                self.all_traces[run_timestamp][func_name] = self.current_trace
                self.current_trace = None
                self.current_key = None
                self.optimized_args = {}

    def _execute_and_trace(self, func, func_name, request, lm_function, *args, **kwargs):
        # Get the function signature
        sig = inspect.signature(func)

        # Map positional args to their names and combine with kwargs
        combined_args = {**kwargs}
        arg_names = list(sig.parameters.keys())[: len(args)]
        combined_args.update(zip(arg_names, args))

        # Include default values for missing arguments
        for param in sig.parameters.values():
            if param.name not in combined_args and param.default is not param.empty:
                combined_args[param.name] = param.default

        if func_name in self.optimized_args:
            optimized_args = self.optimized_args[func_name]["args"]
            if "zenbase" in optimized_args:
                request.zenbase = optimized_args["zenbase"]
            optimized_args.pop("zenbase", None)

        # Replace with optimized arguments if available
        if func_name in self.optimized_args:
            optimized_args = self.optimized_args[func_name]["args"]
            combined_args.update(optimized_args)

        combined_args.update(
            {
                "request": request,
            }
        )
        # Capture input arguments in trace_info
        trace_info = {"args": combined_args, "output": None, "request": request, "lm_function": lm_function}

        # Execute the function and capture its output
        output = func(**combined_args)
        trace_info["output"] = output

        # Store the trace information in the current_trace dictionary
        if self.current_trace is not None:
            self.current_trace[func_name] = trace_info

        return output
