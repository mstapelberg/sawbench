#!/usr/bin/env python3
"""Inspect the structure of an HDF5 file."""

import h5py
import numpy as np

def print_structure(name, obj):
    """Recursively print HDF5 structure."""
    indent = '  ' * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: Dataset {obj.shape} {obj.dtype}")
        if obj.size < 20:  # Print small datasets
            print(f"{indent}  Data: {obj[:]}")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}{name}/: Group")
        if 'attrs' in dir(obj) and obj.attrs:
            print(f"{indent}  Attributes:")
            for key, value in obj.attrs.items():
                print(f"{indent}    {key}: {value}")

def inspect_h5_file(filepath):
    """Inspect an HDF5 file structure."""
    print(f"Inspecting: {filepath}\n")
    print("=" * 60)
    
    with h5py.File(filepath, 'r') as f:
        print("\nTop-level keys:", list(f.keys()))
        print("\nFile attributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        print("\nFull structure:")
        print("/")
        f.visititems(print_structure)
        
        print("\n" + "=" * 60)
        print("\nDetailed information for each top-level group:")
        
        def print_group_details(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"\n{name}/:")
                if obj.attrs:
                    print("  Attributes:")
                    for key, value in obj.attrs.items():
                        print(f"    {key}: {value}")
            elif isinstance(obj, h5py.Dataset):
                print(f"\n{name}:")
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  Size: {obj.size}")
                if obj.size > 0 and obj.size < 100:
                    print(f"  Sample values (first 10): {obj[:10] if len(obj.shape) == 1 else 'Multi-dimensional'}")
                elif obj.size > 0:
                    print(f"  Min: {np.min(obj[:])}, Max: {np.max(obj[:])}")
                if obj.attrs:
                    print("  Attributes:")
                    for key, value in obj.attrs.items():
                        print(f"    {key}: {value}")
        
        f.visititems(print_group_details)

if __name__ == "__main__":
    filepath = "/home/myless/Documents/saw_freq_analysis/rawSignal.h5"
    inspect_h5_file(filepath)

