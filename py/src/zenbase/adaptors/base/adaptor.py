from abc import ABC


class ZenAdaptor(ABC):
    def __init__(self, client=None):
        self.client = client
