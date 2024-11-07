#!/bin/bash

INPUT_ND_CAF_DIR=$1
OUTPUT_ND_CAF_DIR=$2

echo "Running on $(hostname) at ${GLIDEIN_Site}. GLIDEIN_DUNESite = ${GLIDEIN_DUNESite}"

cp -r ${INPUT_TAR_DIR_LOCAL}/* .

# Setup env
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup root v6_28_12 -q e20:p3915:prof
setup ifdhc
python -m venv .venv_3_9_15_torch
source .venv_3_9_15_torch/bin/activate
pip install torch==2.0

# Find and copy over input file
input_file=$(ifdh ls $INPUT_ND_CAF_DIR | head -n $((PROCESS+2)) | tail -n -1)
input_name=${input_file##*/}
input_name=${input_name%.*}
input_name="${input_name}_fdrecpred.root"
echo "Processing ${input_file}"
ifdh cp $input_file $input_name

ls -lrth

# Do the translation
python ndfd_translate_caf.py --vertices_file bin/train_vertices.npy \
                             --model_config bin/models/config_fhc_numu-numu_muresim_oscsampling.json \
                             $input_name \
                             bin/models/model_fhc_numu-numu_muresim_oscsampling.pt

# Copy file back if translation exitted successfully
if [[ $? == 0 ]]
then
  ifdh cp $input_name ${OUTPUT_ND_CAF_DIR}/$input_name
else
  echo "Python script exited badly!"
  echo "Not copying $input_name to $OUTPUT_ND_CAF_DIR"
fi
