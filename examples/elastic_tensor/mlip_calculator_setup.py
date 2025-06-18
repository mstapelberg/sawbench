#!/usr/bin/env python3
"""
MLIP Calculator Setup Guide
===========================

This file shows how to set up different Machine Learning Interatomic Potential
calculators for use with the elastic tensor comparison script.

Replace the EMT calculator in elastic_tensor_comparison.py with one of these.
"""

def setup_mace_calculator():
    """
    Set up MACE calculator
    
    Installation: pip install mace-torch
    """
    try:
        from mace.calculators import MACECalculator
        
        # Option 1: Use pre-trained universal model
        calculator = MACECalculator(
            model_paths=['path/to/your/mace/model.model'],
            device='cpu',  # or 'cuda' if GPU available
            default_dtype='float64'
        )
        
        # Option 2: Use MACE-MP (Materials Project) model
        # calculator = MACECalculator(model_paths=['mace_mp'], device='cpu')
        
        return calculator
        
    except ImportError:
        print("MACE not installed. Install with: pip install mace-torch")
        return None

def setup_nequip_calculator():
    """
    Set up NequIP/Allegro calculator
    
    Installation: pip install nequip
    """
    try:
        from nequip.ase import NequIPCalculator
        
        calculator = NequIPCalculator.from_compiled_model(
            model_path='path/to/your/nequip/model.pth',
            device='cpu'  # or 'cuda'
        )
        
        return calculator
        
    except ImportError:
        print("NequIP not installed. Install with: pip install nequip")
        return None

def setup_chgnet_calculator():
    """
    Set up CHGNet calculator
    
    Installation: pip install chgnet
    """
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        
        # Load pre-trained model
        chgnet = CHGNet.load()
        calculator = CHGNetCalculator(model=chgnet)
        
        return calculator
        
    except ImportError:
        print("CHGNet not installed. Install with: pip install chgnet")
        return None

def setup_m3gnet_calculator():
    """
    Set up M3GNet calculator
    
    Installation: pip install m3gnet
    """
    try:
        from m3gnet.models import M3GNet
        from m3gnet.models import Relaxer
        
        # Load universal potential
        potential = M3GNet.load()
        calculator = potential
        
        return calculator
        
    except ImportError:
        print("M3GNet not installed. Install with: pip install m3gnet")
        return None

def setup_matgl_calculator():
    """
    Set up MatGL calculator (M3GNet successor)
    
    Installation: pip install matgl
    """
    try:
        import matgl
        from matgl.ext.ase import PESCalculator
        
        # Load pre-trained model
        potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        calculator = PESCalculator(potential)
        
        return calculator
        
    except ImportError:
        print("MatGL not installed. Install with: pip install matgl")
        return None

def setup_sevennet_calculator():
    """
    Set up SevenNet calculator
    
    Installation: pip install sevenn
    """
    try:
        from sevenn.sevennet_calculator import SevenNetCalculator
        
        calculator = SevenNetCalculator(
            model_path='path/to/your/sevennet/model.pth',
            device='cpu'
        )
        
        return calculator
        
    except ImportError:
        print("SevenNet not installed. Install with: pip install sevenn")
        return None

def setup_custom_calculator():
    """
    Template for setting up your custom calculator
    """
    # Replace this with your custom calculator setup
    from ase.calculators.emt import EMT  # Example
    
    calculator = EMT()
    
    return calculator

# Usage examples for elastic_tensor_comparison.py:
"""
To use any of these calculators in elastic_tensor_comparison.py, replace this section:

    # For demonstration, using EMT (replace with your MLIP)
    from ase.calculators.emt import EMT
    calculator = EMT()

With one of these:

    # MACE
    calculator = setup_mace_calculator()
    
    # NequIP/Allegro  
    calculator = setup_nequip_calculator()
    
    # CHGNet
    calculator = setup_chgnet_calculator()
    
    # M3GNet
    calculator = setup_m3gnet_calculator()
    
    # MatGL
    calculator = setup_matgl_calculator()
    
    # SevenNet
    calculator = setup_sevennet_calculator()
    
    # Your custom calculator
    calculator = setup_custom_calculator()

Make sure to install the required packages first!
"""

if __name__ == "__main__":
    print("MLIP Calculator Setup Guide")
    print("="*40)
    print("Available calculators:")
    print("1. MACE - Universal ML potential")
    print("2. NequIP/Allegro - Equivariant neural networks") 
    print("3. CHGNet - Crystal graph networks")
    print("4. M3GNet - Materials graph networks")
    print("5. MatGL - M3GNet successor")
    print("6. SevenNet - Seven-body neural network")
    print("\nSee function definitions above for setup instructions.") 