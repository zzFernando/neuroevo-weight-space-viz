from .base import NeuroEvoBase
from .cifar10 import NeuroEvoCIFAR10
from .moons import NeuroEvoMoons

REGISTRY: dict[str, type[NeuroEvoBase]] = {
    "make_moons": NeuroEvoMoons,
    "cifar10":    NeuroEvoCIFAR10,
}

DEFAULTS: dict[str, dict] = {
    "make_moons": {"pop_size": 50,  "n_generations": 80, "hidden_dim": 16, "mutation_rate": 0.11},
    "cifar10":    {"pop_size": 20,  "n_generations": 40, "hidden_dim": 64, "mutation_rate": 0.05},
}

__all__ = [
    "NeuroEvoBase",
    "NeuroEvoMoons",
    "NeuroEvoCIFAR10",
    "REGISTRY",
    "DEFAULTS",
]
