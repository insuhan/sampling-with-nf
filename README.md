# Sampling with Normalizing Flows
Python codes for sampling with normalizing flows

## Install
- This code uses an open-source normalizing flows implementations from https://github.com/karpathy/pytorch-normalizing-flows
- The normalizing flow code (``nflib/flows.py``) is modified for CUDA implementation and dealing with odd hidden dimension case

## Example
```console
$ python run_nf.py --flow_type iaf
```