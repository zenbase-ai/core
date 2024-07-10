from zenbase.adaptors.langfuse_helper.dataset_helper import LangfuseDatasetHelper
from zenbase.adaptors.langfuse_helper.evaluation_helper import LangfuseEvaluationHelper


class ZenLangfuse(LangfuseDatasetHelper, LangfuseEvaluationHelper):
    def __init__(self, client=None):
        LangfuseDatasetHelper.__init__(self, client)
        LangfuseEvaluationHelper.__init__(self, client)

    def get_evaluator(self, data: str):
        data = self.fetch_dataset_demos(data)
        evaluator_kwargs_to_pass = self.evaluator_kwargs.copy()
        evaluator_kwargs_to_pass.update({"data": data})
        return self._metric_evaluator_generator(**evaluator_kwargs_to_pass)
