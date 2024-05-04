#!/bin/bash

INPUT_ND_CAF_DIR=$1
OUTPUT_ND_CAF_DIR=$2

echo "Running on $(hostname) at ${GLIDEIN_Site}. GLIDEIN_DUNESite = ${GLIDEIN_DUNESite}"

cp ${INPUT_TAR_DIR_LOCAL}/* .

ls -lrth

source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup root v6_28_12 -q e20:p3915:prof
setup ifdhc
python -m venv .venv_3_9_15_torch
source .venv_3_9_15_torch/bin/activate
pip install torch==2.0

input_file=$(ifdh ls $INPUT_ND_CAF_DIR | head -n $((PROCESS+2)) | tail -n -1)
input_name=${input_file##*/}
input_name=${input_name%.*}
input_name="${input_name}_fdrecpred.root"
ifdh cp -D $input_file $input_name

python nd_fd_translate_caf.py $input_name model_weights/model.pt

ifdh cp $input_name ${OUTPUT_ND_CAF_DIR}/$input_name
