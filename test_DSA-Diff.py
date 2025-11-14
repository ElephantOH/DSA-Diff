
import argparse
import torch
import numpy as np
import os
import torchvision
from PIL import Image
from tqdm import tqdm
import config as config
from backbones.ncsnpp_generator_adagn import NCSNpp
from dataset import GetDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
        
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
        if self.delta is None:
            self.delta = [args.vp_t]
            for i in range(args.vp_t - 1):
                self.delta.extend([0.0 for _ in range(args.delta_num)])
        return args.delta_range[delta_id] * self.delta[(1 + (t - 1) * args.delta_num) + delta_id] + args.delta_shift[delta_id]


def sample_posterior(pos_coeff, x_0, x_t, t):

    def q_posterior(x_0, x_t, t):
        mean = (
                extract(pos_coeff.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(pos_coeff.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(pos_coeff.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(pos_coeff.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)
        if pos_coeff.delta_var_correction:
            unit_noise = noise / np.linalg.norm(noise.cpu().detach().numpy().flatten())
            noise = unit_noise * pos_coeff.get_delta(t[0], 1)
            if pos_coeff.delta_mean_correction:
                prior = x_0 - mean
                unit_prior = torch.nn.functional.normalize(prior, p=2, dim=1)
                noise += unit_prior * pos_coeff.get_delta(t[0], 0)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise
    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos


def sample_from_model(coefficients, model, x_T, source, args):
    x_t = x_T
    with torch.no_grad():
        for t in reversed(range(args.vp_t)):
            t_module = torch.full((x_t.size(0),), args.vp_sparse * t, dtype=torch.int64).to(x_t.device)
            t_time = torch.full((x_t.size(0),), t, dtype=torch.int64).to(x_t.device)
            latent_z = torch.zeros(x_t.size(0), args.nz, device=x_t.device) if args.sample_fixed else torch.randn(x_t.size(0), args.nz, device=x_t.device)
            h_t = model(torch.cat((x_t, source),axis=1), t_module, latent_z)
            x_td1 = sample_posterior(coefficients, h_t, x_t, t_time)
            x_t = x_td1.detach()
    return x_t

def load_checkpoint(checkpoint_dir, mapping_network, name_of_network, epoch,device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint
    for key in list(ckpt.keys()):
         ckpt[key[7:]] = ckpt.pop(key)
    mapping_network.load_state_dict(ckpt)
    mapping_network.eval()

def evaluate_samples(real_data, fake_sample):
    to_range_0_1 = lambda x: (x + 1.) / 2.
    real_data = real_data.cpu().numpy()
    fake_sample = fake_sample.cpu().numpy()
    psnr_list = []
    ssim_list = []
    mae_list = []
    for i in range(real_data.shape[0]):
        real_data_i = real_data[i]
        fake_sample_i = fake_sample[i]
        real_data_i = to_range_0_1(real_data_i)
        real_data_i = real_data_i / real_data_i.max()
        fake_sample_i = to_range_0_1(fake_sample_i)
        fake_sample_i = fake_sample_i / fake_sample_i.max()
        psnr_val = psnr(real_data_i, fake_sample_i, data_range=real_data_i.max() - real_data_i.min())
        mae_val = np.mean(np.abs(real_data_i - fake_sample_i))
        if args.input_channels == 1:
            ssim_val = ssim(real_data_i[0], fake_sample_i[0], data_range=real_data_i.max() - real_data_i.min())
        elif args.input_channels == 3:
            real_data_i = np.squeeze(real_data_i).transpose(1, 2, 0)
            fake_sample_i = np.squeeze(fake_sample_i).transpose(1, 2, 0)
            ssim_val = ssim(real_data_i, fake_sample_i, channel_axis=-1, data_range=real_data_i.max() - real_data_i.min())
        else:
            raise ValueError("Unsupported number of input channels")
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val * 100)
        mae_list.append(mae_val)
    return psnr_list, ssim_list, mae_list

def save_image(img, save_dir, phase, iteration, input_channels):
    file_path = '{}/{}({}).png'.format(save_dir, phase, str(iteration).zfill(4))
    to_range_0_1 = lambda x: (x + 1.) / 2.
    img = to_range_0_1(img)
    if input_channels == 1:
        torchvision.utils.save_image(img, file_path)
    elif input_channels == 3:
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        image = Image.fromarray(img)
        image.save(file_path)


#%% MAIN FUNCTION
def sample_and_test(args):
    torch.manual_seed(42)

    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))
    epoch_chosen=args.which_epoch

    test_dataset = GetDataset("test", args.input_path, args.source, args.target, dim=args.input_channels, normed=args.normed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = NCSNpp(args).to(device)

    checkpoint_file = args.checkpoint_path + "/{}_{}.pth"
    load_checkpoint(checkpoint_file, model,'EDDM', epoch=str(epoch_chosen), device=device)

    pos_coeff = Posterior_Coefficients(device)
    pos_coeff.delta_var_correction = True
    pos_coeff.refresh_delta(args.delta)
         
    save_dir = args.checkpoint_path + "/generated_samples/EDDM_EPOCH({})".format(epoch_chosen)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    PSNR = []
    SSIM = []
    MAE = []
    image_iteration = 0

    progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing", colour='green')
    for iteration, (source_data, target_data) in progress_bar:
        target_data = target_data.to(device, non_blocking=True)
        source_data = source_data.to(device, non_blocking=True)
        if args.input_channels == 3:
            target_data = target_data.squeeze(1)
            source_data = source_data.squeeze(1)
        x_T = torch.randn_like(target_data)
        fake_sample = sample_from_model(pos_coeff, model, x_T, source_data, args)
        psnr_list, ssim_list, mae_list = evaluate_samples(target_data, fake_sample)
        PSNR.extend(psnr_list)
        SSIM.extend(ssim_list)
        MAE.extend(mae_list)
        progress_bar.set_postfix(PSNR=sum(PSNR) / len(PSNR), SSIM=sum(SSIM) / len(SSIM), MAE=sum(MAE) / len(MAE))
        # tqdm.write(f"[{iteration}/{len(test_dataloader)}] PSNR: {psnr_list[0]}, SSIM: {ssim_list[0]}, MAE: {mae_list[0]}")
        for i in range(fake_sample.shape[0]):
            save_image(fake_sample[i], save_dir, args.phase, image_iteration, args.input_channels)
            image_iteration += 1

    print('TEST PSNR: mean:' + str(sum(PSNR) / len(PSNR)) + ' max:' + str(max(PSNR)) + ' min:' + str(min(PSNR)) + ' var:' + str(np.var(PSNR)))
    print('TEST SSIM: mean:' + str(sum(SSIM) / len(SSIM)) + ' max:' + str(max(SSIM)) + ' min:' + str(min(SSIM)) + ' var:' + str(np.var(SSIM)))
    print('TEST MAE: mean:' + str(sum(MAE) / len(MAE)) + ' max:' + str(max(MAE)) + ' min:' + str(min(MAE)) + ' var:' + str(np.var(MAE) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Diffusion Parameters')
    parser = config.load_config(parser)
    args = parser.parse_args()
    sample_and_test(args)
    
