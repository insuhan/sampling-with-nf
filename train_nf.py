import os
import sys
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform

from utils import sampling_simplex
from nflib.flows import (
        AffineConstantFlow, ActNorm, AffineHalfFlow, 
        SlowMAF, MAF, IAF, Invertible1x1Conv,
        NormalizingFlow, NormalizingFlowModel)

torch.set_default_dtype(torch.float64)


class SimplexUniform(object):
    def __init__(self, ndim):
        self.ndim = ndim

    def sample(self, size):
        return sampling_simplex(size, self.ndim+1)[:,:-1]


def get_arguments():
    parser = argparse.ArgumentParser('simplex')
    parser.add_argument('--flow_type', type=str, default='realnvp')
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--dimx', type=int, default=3)
    parser.add_argument('--dimh', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--print_every', type=int, default=100)
    # parser.add_argument('--batch_size_train', type=int, default=128)
    # parser.add_argument('--batch_size_test', type=int, default=64)
    # parser.add_argument('--clip_grad', type=float, default=0)
    # parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    for n_, v_ in args.__dict__.items():
        print(f"{n_:<10} : {v_}")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    dimx = args.dimx
    depth = args.depth
    max_iters = args.max_iters

    if args.flow_type == 'realnvp':
        flows = [AffineHalfFlow(dim=dimx, nh=args.dimh, parity=i%2) for i in range(depth)]
        norms = [ActNorm(dim=dimx) for _ in flows]
        flows = list(itertools.chain(*zip(norms, flows)))
    elif args.flow_type == 'iaf':
        # IAF (with MADE net, so we get very fast sampling)
        flows = [IAF(dim=dimx, nh=args.dimh, parity=i%2) for i in range(depth)]
        norms = [ActNorm(dim=dimx) for _ in flows]
        flows = list(itertools.chain(*zip(norms, flows)))
    elif args.flow_type == 'maf':
        # MAF (with MADE net, so we get very fast density estimation)
        flows = [MAF(dim=dimx, nh=args.dimh, parity=i%2) for i in range(depth)]
        norms = [ActNorm(dim=dimx) for _ in flows]
        flows = list(itertools.chain(*zip(norms, flows)))
    elif args.flow_type == 'glow':
        # Glow paper
        flows = [Invertible1x1Conv(dim=dimx) for i in range(depth)]
        norms = [ActNorm(dim=dimx) for _ in flows]
        couplings = [AffineHalfFlow(dim=dimx, parity=i%2, nh=args.dimh) for i in range(len(flows))]
        flows = list(itertools.chain(*zip(norms, flows, couplings))) # append a coupling layer after each 1x1
    else:
        raise NotImplementedError

    out_name = f"{args.flow_type}_simplex_dx{dimx}_dh{args.dimh}_depth{depth}.pth"

    prior = TransformedDistribution(Uniform(torch.zeros(dimx).to(device), torch.ones(dimx).to(device)), SigmoidTransform().inv) # Logistic distribution

    model = NormalizingFlowModel(prior, flows).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) # todo tune WD
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters)
    print("number of params: ", sum(p.numel() for p in model.parameters()))

    if not os.path.exists(os.path.join("./models/", args.flow_type)):
        os.makedirs(os.path.join("./models/", args.flow_type))

    sampler = SimplexUniform(dimx)
    model.train()
    pbar = tqdm(range(max_iters))
    loss_all = []

    for k in pbar:
        x = torch.tensor(sampler.sample(128))
        x = x.to(device)

        try:
            zs, prior_logprob, log_det = model(x)
        except:
            try:
                zs, prior_logprob, log_det = model(x)
            except:
                import pdb; pdb.set_trace();
        logprob = prior_logprob + log_det
        loss = -torch.mean(logprob) # NLL
        loss_all.append(loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        pbar.set_description(f"{k:>5} loss: {loss.item():.6f} (avg: {np.mean(loss_all):.6f})")

        if k % 100 == 1:
            print(f"{k:>5} loss: {loss.item():.6f} (avg: {np.mean(loss_all):.6f})")

    torch.save(model.state_dict(), f"./models/{args.flow_type}/{out_name}")

if __name__ == "__main__":
    main()