from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Any


class IDataFrameTransformer(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class IDataFramePipe(ABC):
    def __init__(self, transformers: List[IDataFrameTransformer]) -> None:
        super().__init__()

    @abstractmethod
    def run(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        pass



class IPreProcessor(ABC):
    def __init__(self):
        pass

    def run(self):
        pass
