import inspect
from abc import ABC
from contextlib import contextmanager
from datetime import datetime


class BaseManager(ABC):
    pass


class Zenbase(BaseManager):
    def __init__(self):
        self.all_traces = {}
        self.current_trace = None
        self.current_key = None
        self.optimized_args = {}

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

    def trace_function(self, func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            run_timestamp = f"{func_name}_{datetime.now().isoformat()}"

            if self.current_trace is None:
                with self.trace_context(func_name, run_timestamp):
                    return self._execute_and_trace(func, func_name, *args, **kwargs)
            else:
                return self._execute_and_trace(func, func_name, *args, **kwargs)

        return wrapper

    def _execute_and_trace(self, func, func_name, *args, **kwargs):
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

        # Replace with optimized arguments if available
        if func_name in self.optimized_args:
            optimized_args = self.optimized_args[func_name]["args"]
            combined_args.update(optimized_args)

        # Capture input arguments in trace_info
        trace_info = {"args": combined_args, "output": None}

        # Execute the function and capture its output
        output = func(**combined_args)
        trace_info["output"] = output

        # Store the trace information in the current_trace dictionary
        if self.current_trace is not None:
            self.current_trace[func_name] = trace_info

        return output
