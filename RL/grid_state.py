from dataclasses import dataclass
from typing import Optional
from state import BaseState


@dataclass(frozen=True)
class GridState(BaseState):
    """Grid environment state representation"""
    row: int
    col: int
    trans_state: str
    reward_state: str

    def __hash__(self):
        return hash((self.row, self.col, self.trans_state, self.reward_state))

    def to_tuple(self):
        """Convert to tuple for compatibility"""
        return (self.row, self.col, self.trans_state, self.reward_state)
