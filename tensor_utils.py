"""
Utility functions for creating and manipulating neural data tensors.
"""

import numpy as np


def get_max_channel_count(parsed_data_dict):
    """
    Finds the maximum number of channels across all patients.
    
    Args:
        parsed_data_dict (list): List of patient data dictionaries
        
    Returns:
        int: Maximum number of channels
    """
    all_channel_counts = []
    for patient_data in parsed_data_dict:
        c = len(patient_data['ycoords'])
        all_channel_counts.append(c)
    return max(all_channel_counts)


def replace_nans(array, fill_value=0.0):
    """
    Replaces NaNs in the input array with fill_value.
    
    Args:
        array (np.ndarray): Input array
        fill_value (float): Value to replace NaNs with
        
    Returns:
        np.ndarray: Array with NaNs replaced
    """
    return np.nan_to_num(array, nan=fill_value)


def pad_matrix_to_size(matrix, size, pad_value=0.0):
    """
    Pads a (C, C) matrix to (size, size) with pad_value if C < size.
    If C == size, it just returns the original matrix (with any NaNs replaced).
    
    Args:
        matrix (np.ndarray): Input matrix
        size (int): Target size
        pad_value (float): Value to pad with
        
    Returns:
        np.ndarray: Padded matrix
    """
    matrix = replace_nans(matrix, fill_value=pad_value)
    C = matrix.shape[0]
    
    if C == size:
        return matrix
    else:
        # Create a new (size x size) array filled with pad_value
        padded = np.full((size, size), pad_value, dtype=matrix.dtype)
        # Copy the original matrix into the top-left corner
        padded[:C, :C] = matrix
        return padded


def get_patient_tensors(patient_data, maxC, pad_value=0.0):
    """
    Given a single patient's data dictionary and the max channel count,
    returns:
       phase_magnitude_tensor -> (3, maxC, maxC)
       phase_angle_tensor     -> (3, maxC, maxC)
       comod_tensor           -> (75, maxC, maxC)
    
    Args:
        patient_data (dict): Single patient data dictionary
        maxC (int): Maximum channel count to pad to
        pad_value (float): Value to use for padding
        
    Returns:
        tuple: (phase_magnitude_tensor, phase_angle_tensor, comod_tensor)
    """
    # 1) Phase magnitudes (3 keys: LFPphasemagnitude, LFGphasemagnitude, CSDphasemagnitude)
    phase_mags = []
    for key in ['LFPphasemagnitude', 'LFGphasemagnitude', 'CSDphasemagnitude']:
        # Replace nans and pad
        if key in patient_data:
            mat = pad_matrix_to_size(patient_data[key], maxC, pad_value=pad_value)
            phase_mags.append(mat)
        else:
            # If missing, create a blank
            phase_mags.append(np.full((maxC, maxC), pad_value))
    
    phase_magnitude_tensor = np.stack(phase_mags, axis=0)  # shape: (3, maxC, maxC)

    # 2) Phase angles (3 keys: LFPphaseangle, LFGphaseangle, CSDphaseangle)
    phase_angles = []
    for key in ['LFPphaseangle', 'LFGphaseangle', 'CSDphaseangle']:
        if key in patient_data:
            mat = pad_matrix_to_size(patient_data[key], maxC, pad_value=pad_value)
            phase_angles.append(mat)
        else:
            phase_angles.append(np.full((maxC, maxC), pad_value))
    
    phase_angle_tensor = np.stack(phase_angles, axis=0)  # shape: (3, maxC, maxC)

    # 3) Comodulation (3 sets: lfpLFPcoModCorrs, lfpLFGcoModCorrs, lfpCSDcoModCorrs)
    #    each is 5x5 of (C, C) -> total 25 (C, C) per key -> total 75
    comod_matrices = []
    for key in ['lfpLFPcoModCorrs', 'lfpLFGcoModCorrs', 'lfpCSDcoModCorrs']:
        if key in patient_data:
            # Each key is a 5x5 array of (C, C) mats
            for i in range(5):
                for j in range(5):
                    mat = pad_matrix_to_size(patient_data[key][i][j], maxC, pad_value=pad_value)
                    comod_matrices.append(mat)
        else:
            # If missing, fill with 25 blank mats
            for _ in range(25):
                comod_matrices.append(np.full((maxC, maxC), pad_value))

    comod_tensor = np.stack(comod_matrices, axis=0)  # shape: (75, maxC, maxC)

    return phase_magnitude_tensor, phase_angle_tensor, comod_tensor


def round_up_to_multiple(x, base=8):
    """
    Round up to the nearest multiple of base.
    
    Args:
        x (int or float): Value to round up
        base (int): Base to round to
        
    Returns:
        int: Rounded value
    """
    return int(base * np.ceil(x / base))


def build_all_patient_tensors(parsed_data_dict, pad_value=0.0):
    """
    Main function to:
      1) Find max channel count
      2) For each patient, pad/replace nans
      3) Stack them into consistent shape
      
    Args:
        parsed_data_dict (list): List of patient data dictionaries
        pad_value (float): Value to use for padding
        
    Returns:
        tuple: (all_patient_tensors, paddedC)
            all_patient_tensors: List of (phase_mag_tensor, phase_angle_tensor, comod_tensor) tuples
            paddedC: The padded channel count (multiple of 8)
    """
    maxC = get_max_channel_count(parsed_data_dict)
    paddedC = round_up_to_multiple(maxC, base=8)

    print(f"Max channel count across all patients is {maxC}")
    print(f"Padded to {paddedC} (multiple of 8)")

    all_patient_tensors = []
    for patient_data in parsed_data_dict:
        pmag_tensor, pang_tensor, comod_tensor = get_patient_tensors(patient_data, paddedC, pad_value=pad_value)
        all_patient_tensors.append((pmag_tensor, pang_tensor, comod_tensor))

    return all_patient_tensors, paddedC
