from abc import abstractmethod

from zenbase.helpers.base.adaptor import ZenAdaptor


class BaseEvaluationHelper(ZenAdaptor):
    @abstractmethod
    def set_evaluator_kwargs(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def get_evaluator(self, *args, **kwargs): ...

    @classmethod
    @abstractmethod
    def metric_evaluator(cls, threshold: float = 0.5, **evaluate_kwargs): ...

    # TODO: Should remove and deprecate
