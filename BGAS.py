
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from tqdm import tqdm
import config as config
from backbones.ncsnpp_generator_adagn import NCSNpp
from dataset import GetDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

#%% Bayesian
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import qmc

class BayesianOptimizer:
    def __init__(self, n_params, eval_function, dataloader, model, pos_coeff, device, init_samples=5, max_iter=100, acq_func='EI', kappa=2.0, min_bound=-1, max_bound=1, eps=1e-4):
        self.n_params = n_params
        self.eval_function = eval_function
        self.init_samples = init_samples
        self.max_iter = max_iter
        self.acq_func = acq_func
        self.kappa = kappa
        self.X = []
        self.y = []
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.dataloader = dataloader
        self.model = model
        self.pos_coeff = pos_coeff
        self.device = device
        self.bounds = [(min_bound, max_bound) for _ in range(n_params)]
        self.pos_coeff.delta_mean_correction = False
        self.pos_coeff.delta_var_correction = False
        self.org_v = self.eval_function(self.dataloader, self.model, self.pos_coeff, self.device)
        self.pos_coeff.delta_mean_correction = False
        self.pos_coeff.delta_var_correction = True
        print(f"org_v: {self.org_v}")
        self.kernel = C(
            constant_value=1.0,
            constant_value_bounds=(1e-9, 1e9)
        ) * RBF(
            length_scale=np.ones(n_params),
            length_scale_bounds=(1e-9, 1e9)
        )
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6, normalize_y=True)

    from scipy.stats import qmc

    def _initialize_samples(self, lhs=True):
        if lhs:
            sampler = qmc.LatinHypercube(d=self.n_params)
            samples = sampler.random(n=self.init_samples)
            minb = [self.min_bound for _ in range(self.n_params)]
            maxb = [self.max_bound for _ in range(self.n_params)]
            self.X = qmc.scale(samples, minb, maxb)
        else:
            self.X = np.random.uniform(self.min_bound, self.max_bound, (self.init_samples, self.n_params))

        for x in tqdm(self.X, desc="Initializing samples", total=self.init_samples):
            new_x = [args.vp_t]
            new_x.extend(x)
            print(new_x)
            self.pos_coeff.refresh_delta(new_x)
            self.y.append(self.eval_function(self.dataloader, self.model, self.pos_coeff, self.device))

    def _acquisition_function(self, x):
        x = x.reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)
        y_best = np.max(self.y)
        if self.acq_func == 'EI':
            z = (mu - y_best) / sigma
            ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei
        elif self.acq_func == 'UCB':
            return -(mu + self.kappa * sigma)

    def optimize(self):
        self._initialize_samples()
        self.gp.fit(self.X, self.y)

        total_xy = []

        for i in range(self.max_iter):
            res = minimize(
                fun=self._acquisition_function,
                x0=np.random.uniform(0, 1, self.n_params),
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            x_next = res.x
            x_next = [abs(x) for x in x_next]
            new_x = [args.vp_t]
            new_x.extend(x_next)
            self.pos_coeff.refresh_delta(new_x)
            y_next = self.eval_function(self.dataloader, self.model, self.pos_coeff, self.device)

            self.X = np.vstack([self.X, x_next])
            self.y.append(y_next)
            self.gp.fit(self.X, self.y)
            print(f"Iter {i + 1}: New y = {y_next:.4f}, Current Best = {np.max(self.y):.4f}, Org y = {self.org_v:.4f}")

        best_idx = np.argmax(self.y)
        print(f"Best y = {self.y[best_idx]:.4f}, Orginal y = {self.org_v:.4f}")

        return self.X[best_idx], self.y[best_idx]


#%% Diffusion coefficients
def var_func_affine_logistic(t, vp_max, vp_k1, vp_k2):
    vp_k1 = torch.tensor(vp_k1)
    vp_k2 = torch.tensor(vp_k2)
    log_mean_coeff = -vp_max / vp_k1 * (torch.log(torch.exp(vp_k1 * t - vp_k2) + 1.) - torch.log(torch.exp(-vp_k2) + 1))
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def extract_num(num, shape, device):
    num = torch.tensor([num] * shape[0]).to(device)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    num = num.reshape(*reshape)
    return num


def get_sigma_schedule(vp_t, device):
    eps_small = 1e-3
    t = np.arange(0, vp_t + 1, dtype=np.float64)
    t = t / vp_t
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    var = var_func_affine_logistic(t, args.vp_max, args.vp_k, args.vp_k)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, device, delta=None):
        _, _, self.betas = get_sigma_schedule(args.vp_t, device=device)
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.delta_mean_correction = False
        self.delta_var_correction = False
        if delta is None:
            self.delta = [args.vp_t]
            for i in range(args.vp_t - 1):
                self.delta.extend([0.0 for _ in range(args.delta_num)])
        else:
            self.delta = delta

    def refresh_delta(self, delta):
        assert delta is not None
        self.delta = delta

    def get_delta(self, t, delta_id):
        if t <= 0:
            return 0.0
        assert self.delta is not None
        return args.delta_range[delta_id] * self.delta[(1 + (t - 1) * args.delta_num) + delta_id] + args.delta_shift[delta_id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Diffusion Parameters')
    parser = config.load_config(parser)
    args = parser.parse_args()

    
