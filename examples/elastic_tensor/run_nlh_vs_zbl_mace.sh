#!/bin/bash
# Example script to run NLH vs ZBL comparison with MACE
# 
# Usage: ./run_nlh_vs_zbl_mace.sh

# Set paths - MODIFY THESE PATHS TO MATCH YOUR SYSTEM
#NLH_MODEL="/path/to/ca_lmax2_nlayers2_mlp512_nlh_epoch169.model"
#ZBL_MODEL="/path/to/ca_lmax2_nlayers2_mlp512_zbl_epoch205.model"
NLH_MODEL="/home/myless/Packages/sawbench/examples/data/potentials/gen_6_model_0_L1_isolated-2026-01-16-finetuned_fp64_nh10000_lr1e-4_stagetwo.model"
ZBL_MODEL="/home/myless/Packages/sawbench/examples/data/potentials/gen_8_model_0_L1_isolated-2026-07-29-finetuned.model"
EXP_HDF5="/home/myless/Documents/saw_freq_analysis/fftData.h5"  # Optional

# Run the comparison
python nlh_vs_zbl_comparison_mace.py \
    --nlh-model "$NLH_MODEL" \
    --zbl-model "$ZBL_MODEL" \
    --device cuda \
    --saw-angle-deg 135 \
    --wavelength 8.8e-6 \
    --exp-hdf5 "$EXP_HDF5" \
    --exp-min-mhz 200 \
    --exp-max-mhz 500 \
    --elastic-only

echo "MACE comparison complete! Check results/ directory for output files."
