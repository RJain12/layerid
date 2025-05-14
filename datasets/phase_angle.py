"""
Dataset for phase angle features.
"""

import torch
from datasets.base import BaseChannelLevelDataset
from utils import get_channel_layer


class ChannelLevelPhaseAngleDataset(BaseChannelLevelDataset):
    """
    Dataset for channel-level phase angle data.
    """
    
    def __init__(self, all_tensors, parsed_data_dict, pad_value=99.0):
        """
        Initialize the phase angle dataset.
        
        Args:
            all_tensors (list): List of tensors for all patients
            parsed_data_dict (list): List of patient data dictionaries
            pad_value (float): Value used for padding
        """
        super().__init__(all_tensors, parsed_data_dict, pad_value=pad_value, tensor_index=1)
    
    def _process_channels(self, tensor, mask_tensor, ycoords, dredge):
        """
        Process each channel and add to dataset.
        
        Args:
            tensor (torch.Tensor): Phase angle tensor
            mask_tensor (torch.Tensor): Mask tensor
            ycoords (np.ndarray): Y-coordinates for channels
            dredge (np.ndarray): Layer borders
        """
        for ch in range(len(ycoords)):
            layer = get_channel_layer(ycoords[ch], dredge)
            if layer == -1:
                continue  # Skip extraneous channels
            
            x = tensor[:, ch, :].unsqueeze(1)  # (3, 1, maxC)
            m = mask_tensor[:, ch, :].unsqueeze(1)
            self.inputs.append(x)
            self.masks.append(m)
            self.labels.append(layer)
