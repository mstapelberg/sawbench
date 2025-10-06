#!/bin/bash
# Example script to run NLH vs ZBL comparison
# 
# Usage: ./run_nlh_vs_zbl.sh

# Set paths - MODIFY THESE PATHS TO MATCH YOUR SYSTEM
#NLH_MODEL="/home/myless/Packages/sawbench/examples/data/potentials/config_aware_vs_non_test/ca_lmax2_nlayers2_mlp512_nlh_epoch169.nequip.zip"
#ZBL_MODEL="/home/myless/Packages/sawbench/examples/data/potentials/config_aware_vs_non_test/ca_lmax2_nlayers2_mlp512_zbl_epoch205.nequip.zip"
NLH_MODEL="/home/myless/Packages/sawbench/examples/data/potentials/config_aware_vs_non_test/catw_lmax2_nlayers2_mlp512_nlh_epoch128.nequip.zip"
ZBL_MODEL="/home/myless/Packages/sawbench/examples/data/potentials/config_aware_vs_non_test/mse_lmax2_nlayers2_mlp512_zbl_epoch107.nequip.zip"
#ZBL_MODEL="/home/myless/Packages/sawbench/examples/data/potentials/Nequip_MPTrj_SK_16_06_25.nequip.zip"
EXP_HDF5="/home/myless/Documents/saw_freq_analysis/fftData.h5"  # Optional

# Run the comparison
python nlh_vs_zbl_comparison.py \
    --nlh-model "$NLH_MODEL" \
    --zbl-model "$ZBL_MODEL" \
    --device cuda \
    --saw-angle-deg 135 \
    --wavelength 8.8e-6 \
    --exp-hdf5 "$EXP_HDF5" \
    --exp-min-mhz 200 \
    --exp-max-mhz 500 \
    # --elastic-only

echo "Comparison complete! Check results/ directory for output files."
