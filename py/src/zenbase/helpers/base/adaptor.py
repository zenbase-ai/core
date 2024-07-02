from abc import ABC

from langsmith import Client


class ZenAdaptor(ABC):
    def __init__(self, client=None):
        self.client = client if client else Client()
