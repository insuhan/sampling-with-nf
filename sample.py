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


class SimplexMirrorMap(torch.nn.Module):
    def __init__(self):
        super(SimplexMirrorMap, self).__init__()
        
    def forward(self, x, tmp=None, reverse=False, **kwargs):
        if reverse:
#             jac = torch.diag_embed(x) - x[...,:,None] * x[..., None,:]
#             ld = jac.logdet()
            ld = x.log().sum(-1) + (1 - x.sum(-1)).log()
            return x.exp() / (x.exp().sum(-1, keepdims=True) + 1), ld
        else:
#             jac = torch.diag_embed(1/x_) + (1/(1-x_.sum(-1)))[...,None,None]
#             ld = jac.logdet()
            ld = -x.log().sum(axis=-1) + (1 + x.sum(axis=-1) / (1-x.sum(axis=-1))).log()
            return x.log() - (1 - x.sum(-1, keepdims=True)).log(), ld 


class NFModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(NFModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, x, tmp=None, reverse=False, **kwargs):
        if reverse:
            x_, ld = self.model.flow.backward(x)
            return x_[-1], ld
        else:
            z, _, ld = self.model(x)
            return z[-1], ld


def cross_diff(x1, x2):
    x1 = x1[..., :, None, :]
    x2 = x2[..., None, :, :]
    return x1 - x2


def compute_squared_dist(cross_diff_val):
    return torch.sum(torch.pow(cross_diff_val, 2), axis=-1)


def imq(x1, x2, kernel_width2=None, left_grad=False):
    ret = []
    cross_x1_x2 = cross_diff(x1,x2)
    squared_dists = compute_squared_dist(cross_x1_x2)
    if kernel_width2 is None:
        # kernel_width2 = heuristic_kernel_width(squared_dists)
        n_elems = squared_dists.numel()
        kernel_width2 = torch.min(torch.sort(squared_dists.reshape(-1)).values[n_elems//2:])
    inner = 1. + squared_dists / kernel_width2[..., None, None]
    k = 1. / torch.sqrt(inner)
    ret.append(k)
    grad_x1 = -(inner**(-1.5))[..., None] * cross_x1_x2 / kernel_width2[..., None, None, None]
    ret.append(grad_x1)
    return tuple(ret)

def energy_dist(x, y):
    xx = torch.mean(torch.sqrt(compute_squared_dist(cross_diff(x, x))))
    yy = torch.mean(torch.sqrt(compute_squared_dist(cross_diff(y, y))))
    xy = torch.mean(torch.sqrt(compute_squared_dist(cross_diff(x, y))))
    return 2 * xy - xx - yy


def get_direction(model, alpha, z_, x_=None, kernel=imq):

    n_ = z_.shape[-2]
    if x_ is None:
        x_, _ = model(z_, torch.zeros(z_.shape[0], 1).to(z_), reverse=True)

    logpx = ((alpha[:-1]-1) * x_.log()).sum(-1) + (alpha[-1]-1) * (1 - x_.sum(-1)).log()
    nabla2_psi_theta_inv = torch.cat([
        torch.autograd.grad(x_,
                            z_,
                            torch.eye(z_.shape[-1])[:, i].tile(z_.shape[0], 1).to(x_.device),
                            create_graph=True,
                            retain_graph=True)[0].unsqueeze(0)
        for i in range(z_.shape[-1])
    ]).permute(1, 2, 0)
    
    logdet_dx_dz = nabla2_psi_theta_inv.logdet()
    logqz = logpx + logdet_dx_dz
    grad_logp_eta = torch.autograd.grad(logqz, z_, torch.ones_like(logqz), create_graph=True)[0]

    with torch.no_grad():
        gram, grad_gram = kernel(x_, x_, left_grad=True)
        repulsive_term = torch.einsum("...iab,...ijb->...ja", nabla2_psi_theta_inv, grad_gram) / n_
        weighted_grad = torch.matmul(gram, grad_logp_eta) / n_
        
    return weighted_grad + repulsive_term



def get_arguments():
    parser = argparse.ArgumentParser('simplex')
    parser.add_argument('--flow_type', type=str, default='realnvp')
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--dimx', type=int, default=4)
    parser.add_argument('--dimh', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)
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
    else:
        device = torch.device("cpu")

    dimx = args.dimx
    dimh = args.dimh
    depth = args.depth
    max_iters = args.max_iters
    flow_type = args.flow_type
        
    if flow_type == 'mirror':
        model = SimplexMirrorMap()
    elif flow_type in ['iaf', 'realnvp']:
        from nflib.flows import ActNorm, IAF, NormalizingFlowModel, AffineHalfFlow, MAF
        from torch.distributions import MultivariateNormal
        if flow_type == 'realnvp':
            flows = [AffineHalfFlow(dim=dimx,nh=dimh, parity=i%2) for i in range(depth)]
            norms = [ActNorm(dim=dimx) for _ in flows]
            flows = list(itertools.chain(*zip(norms, flows)))
        elif flow_type == 'iaf':
            # IAF (with MADE net, so we get very fast sampling)
            flows = [IAF(dim=dimx, nh=dimh,parity=i%2) for i in range(depth)]
            norms = [ActNorm(dim=dimx) for _ in flows]
            flows = list(itertools.chain(*zip(norms, flows)))
        elif flow_type == 'maf':
            # MAF (with MADE net, so we get very fast density estimation)
            flows = [MAF(dim=dimx, parity=i%2) for i in range(depth)]
            norms = [ActNorm(dim=dimx) for _ in flows]
            flows = list(itertools.chain(*zip(norms, flows)))

        prior = MultivariateNormal(torch.zeros(dimx).to(device), torch.eye(dimx).to(device))
        model = NormalizingFlowModel(prior, flows).to(device)
        
        try:
            largest_id = max([int(fn_.split("_")[0].split("epoch")[-1]) for fn_ in os.listdir(f"./models/{flow_type}") if f"_{flow_type}_simplex_dx{dimx}_dh{dimh}_depth{depth}" in fn_])
            mname = f"./models/{flow_type}/epoch{largest_id}_{flow_type}_simplex_dx{dimx}_dh{dimh}_depth{depth}.pth"
            print(mname)
            print(model.load_state_dict(torch.load(mname, map_location=device)))
        except:
            import pdb; pdb.set_trace()
            exit(f"File load failed! : {mname}")

        for param in model.parameters():
            param.requires_grad = False

        model = model.to(device)            
        model = NFModelWrapper(model)

    alpha = torch.ones(dimx+1) * 0.1
    alpha[:3] += torch.tensor([90., 5., 5.])
    alpha = alpha.to(device)

    model.model.load_state_dict(torch.load(mname, map_location=device))
    model.model = model.model.to(device)
    for param in model.model.parameters():
        param.requires_grad = False

    n_samples = 20
    x_truth = torch.distributions.dirichlet.Dirichlet(alpha).sample((n_samples,))
    x_truth = x_truth[:,:-1]
    x_truth = x_truth.to(device)

    z_truth, _ = model(x_truth)
    z_truth = z_truth.detach().cpu()
    z_truth = z_truth.to(device)

    n_samples = 10
    z = torch.nn.Parameter(torch.randn((n_samples, dimx)).to(device))

    optimizer = torch.optim.RMSprop([z], lr=args.lr, eps=1e-07, centered=False, alpha=0.9)
    # optimizer = torch.optim.Adam([z], lr=lr)
    optimizer.zero_grad()
    z_all = [z.clone()]
    with torch.no_grad():
        x0 = model(z, torch.zeros(z.shape[0], 1).to(device), reverse=True)[0]
        x_all = [x0.detach().cpu().clone()]
        x0 = x0.to(device)

    x = x0
    eds = []
    pbar = tqdm(range(max_iters))
    for t in pbar:
        z_grad = get_direction(model, alpha, z)
        z.grad = -z_grad
        optimizer.step()
        z_all.append(z.clone())

        x = model(z, torch.zeros(z.shape[0], 1).to(z), reverse=True)[0]
        with torch.no_grad():
            x = x.detach().to(device)
            x_all.append(x.clone())
            ed = energy_dist(x_truth, x)
            eds.append(ed)
            pbar.set_description(f"enery_dist = {ed:10.4f}")
        
    z_all = [z_.detach().cpu() for z_ in z_all]
    x_all = [x_.detach().cpu() for x_ in x_all]


if __name__ == "__main__":
    main()