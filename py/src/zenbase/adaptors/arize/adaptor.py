from zenbase.adaptors.arize.dataset_helper import ArizeDatasetHelper
from zenbase.adaptors.arize.evaluation_helper import ArizeEvaluationHelper


class ZenArizeAdaptor(ArizeDatasetHelper, ArizeEvaluationHelper):
    def __init__(self, client=None):
        ArizeDatasetHelper.__init__(self, client)
        ArizeEvaluationHelper.__init__(self, client)
