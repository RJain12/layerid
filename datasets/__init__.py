"""
Neural data PyTorch dataset module.
Contains dataset classes for different types of neural data.
"""

from .phase_magnitude import ChannelLevelPhaseMagnitudeDataset
from .phase_angle import ChannelLevelPhaseAngleDataset
from .comodulation import ChannelLevelComodDataset

__all__ = [
    'ChannelLevelPhaseMagnitudeDataset',
    'ChannelLevelPhaseAngleDataset',
    'ChannelLevelComodDataset',
]
