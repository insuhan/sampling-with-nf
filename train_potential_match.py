import itertools
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.distributions import MultivariateNormal

from nflib.flows import ActNorm, AffineHalfFlow, MAF, IAF, Invertible1x1Conv, NormalizingFlowModel


def main():

    parser = argparse.ArgumentParser('')
    parser.add_argument('--flow_type', type=str, default='iaf')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--dimx', type=int, default=2)
    parser.add_argument('--dimh', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--mapback', action='store_true', default=False)
    parser.add_argument('--reg', type=float, default=0.01)
    parser.add_argument('--manual_prior', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    args = parser.parse_args()

    if torch.cuda.is_available() and (not args.no_cuda):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dimx = args.dimx
    depth = args.depth
    max_iters = args.max_iters

    out_name = f"{args.flow_type}_push_dx{dimx}_dh{args.dimh}_depth{depth}.pth"
    if not os.path.exists(os.path.join("./models/", args.flow_type)):
        os.makedirs(os.path.join("./models/", args.flow_type))


    cov = torch.randn(args.dimx, args.dimx) / np.sqrt(args.dimx)
    cov = cov @ cov.T
    mu = torch.rand(args.dimx)

    target_dist = MultivariateNormal(mu.to(device), cov.to(device))
    latent_dist = MultivariateNormal(torch.zeros(dimx).to(device), torch.eye(dimx).to(device))

    # RealNVP
    if args.flow_type == 'realnvp':
        flows = [AffineHalfFlow(dim=dimx, nh=args.dimh, parity=i % 2) for i in range(depth)]
        norms = [ActNorm(dim=dimx) for _ in flows]
        flows = list(itertools.chain(*zip(norms, flows)))
    elif args.flow_type == 'iaf':
        # IAF (with MADE net, so we get very fast sampling)
        flows = [IAF(dim=dimx, nh=args.dimh, parity=i % 2) for i in range(depth)]
        norms = [ActNorm(dim=dimx) for _ in flows]
        flows = list(itertools.chain(*zip(norms, flows)))
    elif args.flow_type == 'maf':
        # MAF (with MADE net, so we get very fast density estimation)
        flows = [MAF(dim=dimx, nh=args.dimh, parity=i % 2) for i in range(depth)]
        norms = [ActNorm(dim=dimx) for _ in flows]
        flows = list(itertools.chain(*zip(norms, flows)))
    elif args.flow_type == 'glow':
        # Glow paper
        flows = [Invertible1x1Conv(dim=dimx) for i in range(depth)]
        norms = [ActNorm(dim=dimx) for _ in flows]
        couplings = [AffineHalfFlow(dim=dimx, parity=i % 2, nh=args.dimh) for i in range(len(flows))]
        flows = list(itertools.chain(*zip(norms, flows, couplings)))  # append a coupling layer after each 1x1
    else:
        raise NotImplementedError
    model = NormalizingFlowModel(target_dist, flows).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # todo tune WD
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters)
    print("number of params: ", sum(p.numel() for p in model.parameters()))

    model.train()
    pbar = tqdm(range(max_iters))
    loss_all = []

    for k in pbar:
        x = torch.tensor(torch.randn(128, args.dimx))
        x = x.to(device)

        W_x = latent_dist.log_prob(x)
        _, V_z, log_det = model(x)
        loss = (W_x - V_z - log_det).pow(2).mean()

        if torch.isnan(loss):
            import pdb
            pdb.set_trace()
        loss_all.append(loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()
        sch.step()

        pbar.set_description(f"{k:>5} loss: {loss.item():.6f} (avg: {np.mean(loss_all):.6f})")

        if k % 100 == 1:
            print(f"{k:>5} loss: {loss.item():.6f} (avg: {np.mean(loss_all):.6f})")

        if k % 10000 == 0 and k > 1000:
            torch.save({
                "state_dict": model.state_dict(),
                "loss_all": loss_all,
                "iterations": k
            }, f"./models/{args.flow_type}/potentmatch_{out_name}")

    torch.save({
        "state_dict": model.state_dict(),
        "loss_all": loss_all,
        "iterations": k
    }, f"./models/{args.flow_type}/potentmatch_{out_name}")


if __name__ == '__main__':
    main()
