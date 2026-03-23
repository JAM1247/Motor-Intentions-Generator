
# __init__.py for blocks folder 

"""Signal processing blocks"""

from .generators import BandGenerator, ColoredNoise
from .envelopes import BurstyEnvelope
from .transforms import PACModulate
from .spatial import Reference
from .artifacts import EOGArtifacts, EMGArtifacts, LineNoise, BaselineDrift

__all__ = [
    "BandGenerator", "ColoredNoise",
    "BurstyEnvelope",
    "PACModulate",
    "Reference",
    "EOGArtifacts", "EMGArtifacts", "LineNoise", "BaselineDrift"
]




