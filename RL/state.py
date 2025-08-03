from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class BaseState(ABC):
    """Abstract base class for states"""

    @abstractmethod
    def __hash__(self):
        pass


@dataclass(frozen=True)
class State(BaseState):
    """DBMM state representation"""
    tm_state: str
    rm_state: str
    observation: int

    def __hash__(self):
        return hash((self.tm_state, self.rm_state, self.observation))


@dataclass
class StepResult:
    """Result of an environment step"""
    state: BaseState
    reward: float
    done: bool
    info: dict
