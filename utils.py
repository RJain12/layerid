"""
Common utility functions used across modules.
"""

import numpy as np
import torch


def get_channel_layer(ycoord, dredge_borders):
    """
    Map a ycoord to a layer index [0..7] or -1 if out of range.
    
    Args:
        ycoord (float): Y-coordinate value
        dredge_borders (list or np.ndarray): List of border positions
        
    Returns:
        int: Layer index (0-7) or -1 if out of range
    """
    for i, (start, end) in enumerate(zip(dredge_borders[:-1], dredge_borders[1:])):
        if start >= ycoord > end:
            return i
    return -1  # Extraneous
