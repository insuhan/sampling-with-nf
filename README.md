# Sampling with Normalizing Flows
Python codes for sampling with normalizing flows

## Install
- This code requires open source for normalizing flows implementations (https://github.com/karpathy/pytorch-normalizing-flows)
- Some normalizing flow code (``nflib/flows.py``) are modified to fit the CUDA implementation and odd dimension case

## Run
- Example:
```console
$ python run_nf.py --flow_type iaf
```