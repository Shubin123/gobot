"""Go Bot Training and Curriculum Pipeline Package."""

from .sgf_parser import SGFParser, SGFGame
from .dataset import GoDataset, GoDataPoint
from .curriculum import TrainingCurriculum, CurriculumConfig
from .trainer import GoTrainer
from .self_play import SelfPlayWorker

__all__ = [
    "SGFParser",
    "SGFGame",
    "GoDataset",
    "GoDataPoint",
    "TrainingCurriculum",
    "CurriculumConfig",
    "GoTrainer",
    "SelfPlayWorker",
]
