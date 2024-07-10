from zenbase.adaptors.lunary.dataset_helper import LunaryDatasetHelper
from zenbase.adaptors.lunary.evaluation_helper import LunaryEvaluationHelper


class ZenLunary(LunaryDatasetHelper, LunaryEvaluationHelper):
    def __init__(self, client=None):
        LunaryDatasetHelper.__init__(self, client)
        LunaryEvaluationHelper.__init__(self, client)

    def get_evaluator(self, data: str):
        data = self.fetch_dataset_demos(data)
        evaluator_kwargs_to_pass = self.evaluator_kwargs.copy()
        evaluator_kwargs_to_pass.update({"data": data})
        return self._metric_evaluator_generator(**evaluator_kwargs_to_pass)
