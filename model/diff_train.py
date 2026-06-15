import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat
import math
import random
import matplotlib.pyplot as plt
import argparse
import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import sys
import os
from torch.optim.lr_scheduler import StepLR

from .diff_scheduler import NoiseScheduler
from preprocess.utils import mask_tensor_with_masks
import torch.nn.functional as F



class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        abs_error = torch.abs(input - target)
        is_small_error = abs_error <= self.delta
        small_error_loss = 0.5 * abs_error ** 2
        large_error_loss = self.delta * abs_error - 0.5 * self.delta ** 2
        return torch.where(is_small_error, small_error_loss, large_error_loss).mean()

class diffusion_loss(nn.Module):
    def __init__(self, penalty_factor=1.0, delta=1.0):
        super(diffusion_loss, self).__init__()
        self.mse = nn.MSELoss()
        self.huber_loss = HuberLoss(delta=delta)
        self.penalty_factor = penalty_factor

    def forward(self, y_pred_0, y_true_0, y_pred_1, y_true_1):
        loss_mse = self.mse(y_pred_0, y_true_0)
        loss_huber = self.huber_loss(y_pred_1, y_true_1) * self.penalty_factor  
        return loss_mse + loss_huber
        # return loss_mse


def split_integer_exp_decay(S, ar_step_decay=1.0):
    if ar_step_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ar_step_decay) / (1 - math.pow(ar_step_decay, S))
        p = [base * math.pow(ar_step_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]

    
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def get_ar_weights(split_sizes, cumsum, max_weight=2.0, min_weight=1.0, schedule="linear"):
    assert max_weight >= min_weight
    if max_weight == min_weight:
        return torch.tensor(min_weight)

    weights = []
    full_len = cumsum[-1]
    if schedule == "cosine":
        for size, x in zip(split_sizes, cumsum[:-1]):
            weights.extend(
                size * [
                    math.cos(math.pi / full_len * x) * (max_weight - min_weight) / 2 + 
                    (max_weight + min_weight) / 2
                ]
            )
    elif schedule == "linear":
        for size, x in zip(split_sizes, cumsum[:-1]):
            weights.extend(size * [max_weight - (max_weight - min_weight) / full_len * x])
    else:
        raise NotImplementedError

    return torch.tensor(weights)


def get_attn_mask(sample_len, cond_len, split_sizes=None, cumsum=None):
    visiable_len = sample_len - split_sizes[-1]
    ctx_len = cond_len + visiable_len
    seq_len = ctx_len + sample_len

    attn_mask = torch.ones(size=(seq_len, seq_len))
    attn_mask[:, :cond_len] = 0

    # build `triangle` masks
    triangle1 = torch.ones(size=(visiable_len, visiable_len))
    triangle2 = torch.ones(size=(sample_len, visiable_len))
    triangle3 = torch.ones(size=(sample_len, sample_len))
    for i in range(len(split_sizes) - 1):
        triangle1[cumsum[i]:cumsum[i+1], 0:cumsum[i+1]] = 0
        triangle2[cumsum[i+1]:cumsum[i+2], 0:cumsum[i+1]] = 0
    for i in range(len(split_sizes)):
        triangle3[cumsum[i]:cumsum[i+1], cumsum[i]:cumsum[i+1]] = 0

    # copy mask to attention mask
    attn_mask[cond_len:ctx_len, cond_len:ctx_len] = triangle1
    attn_mask[ctx_len:, cond_len:ctx_len] = triangle2
    attn_mask[ctx_len:, ctx_len:] = triangle3

    return attn_mask[None, None, :, :]




class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, input_options=[4784, 6143], Sigmoid=1):
        super(VAE, self).__init__()
        self.input_options = input_options
        
        # Two different linear layers for the two possible input shapes.
        self.lin1 = nn.Linear(input_options[0], hidden_dim)  # Encoder 1 input layer (with variation)
        self.lin2 = nn.Linear(input_options[1], hidden_dim)  # Encoder 2 input layer (without variation)
        
        # Encoder 1: Outputs both mu and logvar for the latent distribution (with variation)
        self.encoder1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2 * latent_dim)  # Outputs mu and logvar for reparameterization
        )
        
        # Encoder 2: Outputs latent features without mu and logvar (without variation)
        self.encoder2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)  # Outputs only latent features without variation
        )
        
        # Decoder: Use Sigmoid on the output if Sigmoid == 1, else no Sigmoid.
        if Sigmoid == 1:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, sigma^2)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        # Determine which encoder to use based on input shape
        if x.shape[1] == self.input_options[0]:
            x = self.lin1(x)  # Pass through lin1 for encoder1
            latent_params = self.encoder1(x)  # Pass through encoder1 (with variation)
            mu, logvar = torch.chunk(latent_params, 2, dim=-1)  # Split into mu and logvar
            return mu, logvar
        elif x.shape[1] == self.input_options[1]:
            x = self.lin2(x)  # Pass through lin2 for encoder2
            latent_params = self.encoder2(x)  # Pass through encoder2 (without variation)
            return latent_params, None  # Only latent features, no mu and logvar

    def sample(self, x):
        mu, logvar = self.encode(x)
        if logvar is not None:  # If variation exists (encoder1)
            z = self.reparameterize(mu, logvar)
        else:  # If no variation (encoder2)
            z = mu  # No reparameterization needed
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        if logvar is not None:  # If variation exists
            z = self.reparameterize(mu, logvar)
        else:  # If no variation
            z = mu  # No reparameterization needed
        decoded = self.decode(z)
        return decoded, mu, logvar




def normal_train_diff(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda:0'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 mask_nonzero_ratio= None,
                 mask_zero_ratio = None,
                 vae=VAE,
                 ar_step_decay=None,
                 decoder_train=None
                 ):
    """Generic training function

    Args:
        lr (float):
        momentum (float): momentum
        max_iteration (int, optional): training iteration. Defaults to 30000.
        pred_type (str, optional): Predicted type noise or x_0. Defaults to 'noise'.
        batch_size (int, optional): Defaults to 1024.
        diffusion_step (int, optional): Number of diffusion steps. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:0').
        is_class_condi (bool, optional): Whether to use condition. Defaults to False.
        is_tqdm (bool, optional): Turn on the progress bar. Defaults to True.
        is_tune (bool, optional): Whether to use ray tune. Defaults to False.
        condi_drop_rate (float, optional): Whether to use classifier free guidance to set the drop rate. Defaults to 0..

    Raises:
        NotImplementedError: _description_
    """

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    criterion = diffusion_loss()
    model.to(device)

    if decoder_train==1:
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(vae.parameters()), 
                                      lr=lr, 
                                      weight_decay=0)
    else:
        encoder_params = list(filter(lambda p: p.requires_grad, vae.parameters()))
        optimizer = torch.optim.AdamW(list(model.parameters()) + encoder_params, 
                                      lr=lr, 
                                      weight_decay=0)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    vae.train()
    model.train()
   
    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_hat, x_cond) in enumerate(dataloader):
            x, x_hat, x_cond = x.float().to(device), x_hat.float().to(device), x_cond.float().to(device)
            
            if decoder_train==1:
                x, x_nonzero_mask, x_zero_mask = mask_tensor_with_masks(x, mask_zero_ratio, mask_nonzero_ratio)
                x_hat, x_hat_nonzero_mask, x_hat_zero_mask = mask_tensor_with_masks(x_hat, mask_zero_ratio, mask_nonzero_ratio)

                x_noise = torch.randn(x.shape).to(device)
                x_hat_noise = torch.randn(x_hat.shape).to(device)
                timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
                timesteps = timesteps.to(device)
                x_t = noise_scheduler.add_noise(x,
                                            x_noise,
                                            timesteps=timesteps)

                x_hat_t = noise_scheduler.add_noise(x_hat,
                                            x_hat_noise,
                                            timesteps=timesteps)

                x_noisy = x_t * x_nonzero_mask + x * (1 - x_nonzero_mask)
                x_hat_noisy = x_hat_t * x_hat_nonzero_mask + x_hat * (1 - x_hat_nonzero_mask)
                x=vae.sample(x)
                x_hat=vae.sample(x_hat)
                x_noisy=vae.sample(x_noisy)
            if decoder_train==0:
                x=vae.sample(x)
                x_hat=vae.sample(x_hat)

                x, x_nonzero_mask, x_zero_mask = mask_tensor_with_masks(x, mask_zero_ratio, mask_nonzero_ratio)
                x_hat, x_hat_nonzero_mask, x_hat_zero_mask = mask_tensor_with_masks(x_hat, mask_zero_ratio, mask_nonzero_ratio)

                x_noise = torch.randn(x.shape).to(device)
                x_hat_noise = torch.randn(x_hat.shape).to(device)
                timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
                timesteps = timesteps.to(device)
                x_t = noise_scheduler.add_noise(x,
                                                x_noise,
                                                timesteps=timesteps)

                x_hat_t = noise_scheduler.add_noise(x_hat,
                                                x_hat_noise,
                                                timesteps=timesteps)

                x_noisy = x_t * x_nonzero_mask + x * (1 - x_nonzero_mask)
                x_hat_noisy = x_hat_t * x_hat_nonzero_mask + x_hat * (1 - x_hat_nonzero_mask)
            L = x.shape[1]
            split_sizes, cumsum = split_integer_exp_decay(L, ar_step_decay=0.9)
            attn_mask = get_attn_mask(L, model.num_cond_tokens, split_sizes, cumsum)
            attn_mask = attn_mask.bool().to(x.device)
            ar_weights = get_ar_weights(
                split_sizes,
                cumsum,
                max_weight=2.0,
                min_weight=1.0,
                schedule="linear"
            )
            ar_weights = ar_weights.to(x.device).reshape(1, -1, 1)

            noise_pred = model(xn=x_noisy, t=timesteps, x=x, y=x_hat, attn_mask=attn_mask, last_split_size=split_sizes[-1], noise=x_noise)
            noise_pred=noise_pred.squeeze()
            if decoder_train==1:
                noise_pred=vae.decode(noise_pred)

            loss = criterion(x_noise * x_nonzero_mask, noise_pred * x_nonzero_mask, x_noise * x_zero_mask,
                              noise_pred * x_zero_mask)* ar_weights.mean()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss = epoch_loss / (i + 1)
        
        print("Epoch:", epoch, "Loss:", epoch_loss)
        
        

        current_lr = optimizer.param_groups[0]['lr']

        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}, lr:{current_lr:.2e}')

        if is_tune:
            session.report({'loss': epoch_loss})