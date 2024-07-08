Repository for running nd reco -> fd reco translation on PRISM CAFs.

See [radiradev/dune-near-to-far](https://github.com/radiradev/dune-near-to-far) for training code.

## Setup

On the dgpvms in an SL7 shell. For first setup do:
```
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup root v6_28_12 -q e20:p3915:prof
python -m venv .venv_3_9_15_torch
source .venv_3_9_15_torch/bin/activate
pip install torch==2.0
```

Now you have the virtual environment, just do `source setup.sh`.

## Usage

```
usage: ndfd_translate_caf.py [-h] infile model_weights

positional arguments:
  infile         input ND CAF file for friend FD pred tree to be added to
  model_weights

optional arguments:
  -h, --help     show this help message and exit
```

## Grid jobs

Make a tarball of these files:
```
ndfd_translate_caf.py
model.py
utils.py
model_weights/
```

Submit to the grid with:
```
jobsub_submit -G dune -N <N> --disk=10Gb --memory=6000MB --expected-lifetime=24h --cpu=8 --resource-provides=usage_model=DEDICATED,OPPORTUNISTIC,OFFSITE --tar_file_name=dropbox://<path_to_repo>/jobdata.tar.gz --use-cvmfs-dropbox --singularity-image /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl7:latest --append_condor_requirements='(TARGET.HAS_Singularity==true&&TARGET.HAS_CVMFS_dune_opensciencegrid_org==true&&TARGET.HAS_CVMFS_larsoft_opensciencegrid_org==true&&TARGET.CVMFS_dune_opensciencegrid_org_REVISION>=1105&&TARGET.HAS_CVMFS_fifeuser1_opensciencegrid_org==true&&TARGET.HAS_CVMFS_fifeuser2_opensciencegrid_org==true&&TARGET.HAS_CVMFS_fifeuser3_opensciencegrid_org==true&&TARGET.HAS_CVMFS_fifeuser4_opensciencegrid_org==true)' --lines '+FERMIHTC_AutoRelease=True' --lines '+FERMIHTC_GraceMemory=2048' --lines '+FERMIHTC_GraceLifetime=7200' file://<path_to_repo>/grid/jobsub_submit.sh <input_dir> <output_dir>
```
Replacing `<N>` with the number of ND CAF files in the input directory, and `<path_to_repo>,<input_dir>,<output_dir>` with your paths.

## Notes

- If we want to use different architectures or different ND/FD reco vars the code will need to be updated. We should refactor a bit to accomodate this more easily.
