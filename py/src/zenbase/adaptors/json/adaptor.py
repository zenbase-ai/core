from zenbase.adaptors.json.dataset_helper import JSONDatasetHelper
from zenbase.adaptors.json.evaluation_helper import JSONEvaluationHelper


class JSONAdaptor(JSONDatasetHelper, JSONEvaluationHelper):
    def __init__(self, client=None):
        JSONDatasetHelper.__init__(self, client)
        JSONEvaluationHelper.__init__(self, client)
