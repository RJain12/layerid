"""
Data loading module for neural data processing.
Contains functions to load and convert MATLAB data structures.
"""

import numpy as np
import scipy.io as sio


def load_mat_data(file_path='parsedData.mat'):
    """
    Load MATLAB data file and extract parsed data.
    
    Args:
        file_path (str): Path to the .mat file
        
    Returns:
        list: List of patient data dictionaries
    """
    # Load the .mat file with better struct handling
    mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    parsed_data = mat_data['parsedData']
    
    # Convert the parsed data to Python dictionaries
    parsed_data_dict = [mat_struct_to_dict(patient) for patient in np.ravel(parsed_data)]
    
    return parsed_data_dict


def mat_struct_to_dict(mat_struct):
    """ 
    Recursively convert a MATLAB struct to a nested Python dictionary.
    
    Args:
        mat_struct: MATLAB struct or any other data type
        
    Returns:
        dict or original value: Converted structure or original value if not a struct
    """
    if isinstance(mat_struct, sio.matlab.mio5_params.mat_struct):
        result = {}
        for field in mat_struct._fieldnames:  # Extract field names
            value = getattr(mat_struct, field)  # Access field data
            
            # Recursively process if it's another struct
            if isinstance(value, sio.matlab.mio5_params.mat_struct):
                result[field] = mat_struct_to_dict(value)
            elif isinstance(value, np.ndarray):
                # Check if it's an array of structs
                if value.dtype.names is not None:
                    result[field] = [mat_struct_to_dict(v) for v in value]
                else:
                    result[field] = value  # Keep NumPy arrays as they are
            else:
                result[field] = value  # Assign raw value
        
        return result
    else:
        return mat_struct  # If not a struct, return the raw value
