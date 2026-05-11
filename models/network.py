import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from models.loss import l1_loss


class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet

        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        if phase=='train' or phase=='val':
            pass
        else:
            use_timesteps = space_timesteps(self.beta_schedule['test']['n_timestep'], self.beta_schedule['test']['timestep_respacing'])
            timestep_map = []
            original_num_steps = len(betas)
            alphas = 1. - betas
            gammas = np.cumprod(alphas, axis=0)
            last_gammas = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(gammas):
                if i in use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_gammas)
                    last_gammas = alpha_cumprod
                    timestep_map.append(i)
            betas = np.array(new_betas)
        alphas = 1. - betas
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('gammas_prev', to_torch(gammas_prev))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
                extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
                extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def predict_eps_from_xstart(self, y_t, t, pred_xstart):
        return (extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t
                - pred_xstart) / extract(self.sqrt_recipm1_gammas, t, y_t.shape)

    def q_posterior(self, y_0_hat, y_t, t, y_cond):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
                extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None, hint_tokens=None, hint_input=None,
                        use_mask_guided=True):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
            y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level, hint_tokens=hint_tokens, hint_input=hint_input, use_mask_guided=use_mask_guided))

        if clip_denoised:
            y_0_hat.clamp_(-1, 1)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t, y_cond=y_cond)
        return model_mean, posterior_log_variance, y_0_hat

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
                sample_gammas.sqrt() * y_0 +
                (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None, hint_tokens=None, hint_input=None, use_mask_guided=True):
        model_mean, model_log_variance, y_0_hat = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, hint_tokens=hint_tokens, hint_input=hint_input, use_mask_guided=use_mask_guided)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def ddim_sample(self, y_t, t, clip_denoised=True, y_cond=None, eta=0.0, hint_tokens=None, hint_input=None,
                    use_mask_guided=True):
        # noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        # eps = self.denoise_fn(torch.cat([y_cond, y_t], dim=1),noise_level)
        model_mean, model_log_variance, y_0_hat = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, hint_tokens=hint_tokens, hint_input=hint_input, use_mask_guided=use_mask_guided)

        eps = self.predict_eps_from_xstart(y_t, t, y_0_hat)
        alpha_bar = extract(self.gammas, t, y_t.shape)
        alpha_bar_prev = extract(self.gammas_prev, t, y_t.shape)
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        mean_pred = (
                y_0_hat * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(y_t.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample
        # return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, phase='test', hint_tokens=None, hint_input=None, use_mask_guided=True):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)
        # y_t为重建过程中与SAR图像叠加的噪声图
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            # y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if phase=='test':
                eta = 0
            else:
                eta = 1
            y_t = self.ddim_sample(y_t, t, y_cond=y_cond, eta=eta, hint_tokens=hint_tokens, hint_input=hint_input, use_mask_guided=use_mask_guided)
            # if mask is not None:
            #     y_t = y_0*(1.-mask) + mask*y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None, hint_tokens=None, hint_input=None, use_mask_guided=True):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        # t = torch.arange(1, self.num_timesteps + 1, dtype=torch.long, device=y_0.device).unsqueeze(0).expand(b, -1)
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
        # l1_loss = []
        if mask is not None:
            noise_hat = self.denoise_fn(
                torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1),
                sample_gammas,
                hint_tokens=hint_tokens,
                hint_input=hint_input,
                use_mask_guided=use_mask_guided
            )
            loss = self.loss_fn(mask * noise, mask * noise_hat)
        else:
            # 加入色条的特征计算，输入到u-Net中
            noise_hat = self.denoise_fn(
                torch.cat([y_cond, y_noisy], dim=1),
                sample_gammas,
                hint_tokens=hint_tokens,
                hint_input=hint_input,
                use_mask_guided=use_mask_guided
            )
            loss = self.loss_fn(noise, noise_hat)
        return loss


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3, timestep_respacing=None
                       ):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
        # scale = 1000 / n_timestep
        # beta_start = scale * 0.0001
        # beta_end = scale * 0.02
        # return np.linspace(
        #     beta_start, beta_end, n_timestep, dtype=np.float64
        # )
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)
