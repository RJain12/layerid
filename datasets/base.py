"""
Base dataset class with common utilities for neural data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import get_channel_layer


class BaseChannelLevelDataset(Dataset):
    """
    Base class for channel-level datasets.
    """
    
    def __init__(self, all_tensors, parsed_data_dict, pad_value=99.0, tensor_index=0):
        """
        Initialize the base dataset class.
        
        Args:
            all_tensors (list): List of tensors for all patients
            parsed_data_dict (list): List of patient data dictionaries
            pad_value (float): Value used for padding
            tensor_index (int): Index to access the correct tensor type (0=pmag, 1=pang, 2=comod)
        """
        self.inputs = []
        self.masks = []
        self.labels = []
        self.tensor_index = tensor_index
        self.pad_value = pad_value
        
        self._process_data(all_tensors, parsed_data_dict)
    
    def _process_data(self, all_tensors, parsed_data_dict):
        """
        Process the data and organize it by channel.
        
        Args:
            all_tensors (list): List of tensors for all patients
            parsed_data_dict (list): List of patient data dictionaries
        """
        for idx, tensors in enumerate(all_tensors):
            patient = parsed_data_dict[idx]
            ycoords = np.array(patient['ycoords'])
            dredge = np.array(patient['DREDgeLayerBorders'])
            
            tensor_data = tensors[self.tensor_index]
            tensor = torch.tensor(tensor_data, dtype=torch.float32)
            mask_tensor = (tensor != self.pad_value).float()
            
            self._process_channels(tensor, mask_tensor, ycoords, dredge)
    
    def _process_channels(self, tensor, mask_tensor, ycoords, dredge):
        """
        Process each channel and add to dataset.
        
        Args:
            tensor (torch.Tensor): Data tensor
            mask_tensor (torch.Tensor): Mask tensor
            ycoords (np.ndarray): Y-coordinates for channels
            dredge (np.ndarray): Layer borders
        """
        # To be implemented by subclasses
        pass
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.inputs)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (input, mask, label)
        """
        return self.inputs[idx], self.masks[idx], self.labels[idx]
