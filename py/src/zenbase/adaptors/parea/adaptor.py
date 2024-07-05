__all__ = ["ZenParea"]

from zenbase.adaptors.parea.dataset_helper import PareaDatasetHelper
from zenbase.adaptors.parea.evaluation_helper import PareaEvaluationHelper
from zenbase.optim.metric.types import CandidateEvaluator


class ZenParea(PareaDatasetHelper, PareaEvaluationHelper):
    def __init__(self, client=None):
        PareaDatasetHelper.__init__(self, client)
        PareaEvaluationHelper.__init__(self, client)

    def get_evaluator(self, data: str) -> CandidateEvaluator:
        evaluator_kwargs_to_pass = self.evaluator_kwargs.copy()
        evaluator_kwargs_to_pass.update({"data": self.fetch_dataset_list_of_dicts(data)})
        return self._metric_evaluator_generator(**evaluator_kwargs_to_pass)
