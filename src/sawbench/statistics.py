import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, kurtosis
from typing import Union, List, Tuple, Dict, Any

def calculate_summary_statistics(
    data_array: np.ndarray, 
    data_name: str = "Data", 
    units: str = "MHz"
) -> Dict[str, Union[float, int]]:
    """
    Calculates and prints summary statistics for a given dataset.

    Args:
        data_array (np.ndarray): 1D array of numerical data.
        data_name (str): Descriptive name for the dataset (e.g., "Experimental Frequencies").
        units (str): Units of the data for printing (e.g., "MHz", "Hz").

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing the calculated statistics:
            'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Kurtosis'.
            Returns an empty dict if data_array is empty or all NaNs.
    """
    stats_results = {}
    if data_array is None or data_array.size == 0:
        print(f"{data_name}: No data provided.")
        return stats_results

    valid_data = data_array[~np.isnan(data_array)]
    if valid_data.size == 0:
        print(f"{data_name}: No valid (non-NaN) data points.")
        return stats_results

    stats_results['Count'] = len(valid_data)
    stats_results['Mean'] = np.nanmean(valid_data)
    stats_results['Median'] = np.nanmedian(valid_data)
    stats_results['Std Dev'] = np.nanstd(valid_data)
    stats_results['Min'] = np.nanmin(valid_data)
    stats_results['Max'] = np.nanmax(valid_data)
    stats_results['Kurtosis'] = kurtosis(valid_data, nan_policy='omit')
    
    print(f"\n--- Summary Statistics for {data_name} ({units}) ---")
    print(f"  Count:    {stats_results['Count']}")
    print(f"  Mean:     {stats_results['Mean']:.2f}")
    print(f"  Median:   {stats_results['Median']:.2f}")
    print(f"  Std Dev:  {stats_results['Std Dev']:.2f}")
    print(f"  Min:      {stats_results['Min']:.2f}")
    print(f"  Max:      {stats_results['Max']:.2f}")
    print(f"  Kurtosis: {stats_results['Kurtosis']:.2f}")
    
    return stats_results

def perform_ks_test(
    data1: np.ndarray, 
    data2: np.ndarray, 
    data1_name: str = "Dataset 1", 
    data2_name: str = "Dataset 2"
) -> Tuple[float, float] | Tuple[None, None]:
    """
    Performs a 2-sample Kolmogorov-Smirnov test to compare two distributions.

    Args:
        data1 (np.ndarray): First dataset (1D array).
        data2 (np.ndarray): Second dataset (1D array).
        data1_name (str): Name of the first dataset for printing.
        data2_name (str): Name of the second dataset for printing.

    Returns:
        Tuple[float, float] | Tuple[None, None]: A tuple containing (KS statistic, p-value).
                                                  Returns (None, None) if either dataset is empty
                                                  or contains insufficient valid data.
    """
    if data1 is None or data1.size == 0 or data2 is None or data2.size == 0:
        print("KS Test: One or both datasets are empty. Skipping test.")
        return None, None

    valid_data1 = data1[~np.isnan(data1)]
    valid_data2 = data2[~np.isnan(data2)]

    if valid_data1.size < 2 or valid_data2.size < 2: # KS test needs at least a few points
        print(f"KS Test: Insufficient valid data in one or both datasets ({len(valid_data1)} vs {len(valid_data2)}). Skipping test.")
        return None, None

    print(f"\n--- Kolmogorov-Smirnov Test: {data1_name} vs {data2_name} ---")
    ks_statistic, p_value = ks_2samp(valid_data1, valid_data2)
    print(f"  KS Statistic: {ks_statistic:.4f}")
    print(f"  P-value:      {p_value:.4g}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"  Result: The p-value ({p_value:.4g}) is less than alpha ({alpha}), suggesting the distributions are statistically different.")
    else:
        print(f"  Result: The p-value ({p_value:.4g}) is greater than or equal to alpha ({alpha}), suggesting no statistically significant difference between the distributions.")
        
    return ks_statistic, p_value 