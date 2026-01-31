from abc import ABC, abstractmethod
from typing import Any

from data.schema import UnifiedTrajectory


class BaseAdapter(ABC):
    @abstractmethod
    def can_handle(self, raw_obj: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def convert(self, raw_obj: Any, meta: dict) -> UnifiedTrajectory:
        raise NotImplementedError
