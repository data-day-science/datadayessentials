import numpy as np
from flaml import AutoML


class PatchedAutoML(AutoML):
    classes_ = np.array([0, 1])
    is_estimator = True

    @staticmethod
    def __sklearn_is_fitted__():
        return True

    def set_classes(self):
        self.classes_ = np.array([0, 1])
