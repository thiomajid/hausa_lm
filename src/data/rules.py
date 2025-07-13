import typing as tp
from dataclasses import dataclass

Operand = tp.Literal["eq", "ne", "lt", "le", "gt", "ge"]


@dataclass
class BinaryFilterRule:
    lhs: str
    op: Operand
    rhs: tp.Any

    def as_predicate(self) -> tp.Callable[[dict], bool]:
        match self.op:
            case "eq":
                return lambda ex: ex[self.lhs] == self.rhs
            case "ne":
                return lambda ex: ex[self.lhs] != self.rhs
            case "lt":
                return lambda ex: ex[self.lhs] < self.rhs
            case "le":
                return lambda ex: ex[self.lhs] <= self.rhs
            case "gt":
                return lambda ex: ex[self.lhs] > self.rhs
            case "ge":
                return lambda ex: ex[self.lhs] >= self.rhs
            case _:
                raise ValueError(f"{self.op} is not a supported operand")
