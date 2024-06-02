from typing import Callable, Awaitable


class LMFunction[I, O]:
    def __init__(self, function: LMFunction[I, O]) -> None:
        self.function = function
