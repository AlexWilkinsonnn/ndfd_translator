Repository for running nd reco -> fd reco translation on PRISM CAFs.

See [radiradev/dune-near-to-far](https://github.com/radiradev/dune-near-to-far) for training code.

### Setup

On the dgpvms in an SL7 shell. For first setup do:
```
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup root v6_28_12 -q e20:p3915:prof
python -m venv .venv_3_9_15_torch
source .venv_3_9_15_torch/bin/activate
pip install torch==2.0
```

Now you have the virtual environment, just do `source setup.sh`.
