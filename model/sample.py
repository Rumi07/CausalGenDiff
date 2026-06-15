import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from preprocess.utils import calculate_rmse_per_gene, calculate_pcc_per_gene,calculate_pcc_with_mask,calculate_rmse_with_mask
from preprocess.utils import mask_tensor_with_masks
def model_sample_diff(model, device, dataloader, total_sample, time, is_condi, condi_flag, vae=None):
    noise = []
    i = 0
    for _, x_hat, x_cond in dataloader: 
        x_hat, x_cond = x_hat.float().to(device), x_cond.float().to(device) 
        x=vae.sample(total_sample[i:i+len(x_cond)])
        x_hat=vae.sample(x_hat)
        D = model.hidden_size
        L = model.num_patches
        
        c = model.latent_channels
        p = model.patch_size
        cond_len = model.num_cond_tokens
        patch_dim = c * p ** 2
        N = x_cond.shape[0]
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)
        kv_cache = {i:{"k": None, "v": None} for i in range(len(model.blocks))}

        
        y_embed = model.y_embedder(x_hat)
        y_embed = y_embed.reshape(-1, cond_len, D)
        cond = y_embed + model.cond_pos_embed
        model.forward_cache_update(cond, kv_cache=kv_cache)

      
        if not is_condi:

            n = model(total_sample[i:i+len(x_cond)], t, None) 
        else:
            x = x.reshape(N, -1, patch_dim)
            n= model.forward_inference(xn=x, t=t, 
                                       kv_cache=kv_cache, 
                                       pos_embed=model.pos_embed.repeat(N, 1, 1), cfg_scale=1.0, cfg_interval=[0, 1000])
           
            n=n.squeeze()
            n=vae.decode(n)
           
            
           
        noise.append(n)
        i = i+len(x_cond)
    noise = torch.cat(noise, dim=0)
    return noise

def sample_diff(model,
                dataloader,
                noise_scheduler,
                mask_nonzero_ratio = None,
                mask_zero_ratio = None,
                gt = None,
                sc = None,
                device=torch.device('cuda:0'),
                num_step=1000,
                sample_shape=(7060, 2000),
                is_condi=False,
                sample_intermediate=200,
                model_pred_type: str = 'noise',
                is_classifier_guidance=False,
                omega=0.1,
                is_tqdm = True,
                vae=None):
    model.eval()
    gt = torch.tensor(gt).to(device)
    sc = torch.tensor(sc).to(device)
    x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)
    timesteps = list(range(num_step))[::-1]  
    gt_mask, mask_nonzero, mask_zero = mask_tensor_with_masks(gt, mask_zero_ratio, mask_nonzero_ratio)
    mask = torch.tensor(mask_nonzero).to(device)
   
    x_t = x_t
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
        
            model_output = model_sample_diff(model,
                                        device=device,
                                        dataloader=dataloader,
                                        total_sample=x_t,  # x_t
                                        time=time,  # t
                                        is_condi=is_condi,
                                        condi_flag=True,vae=vae)
            if is_classifier_guidance:
                model_output_uncondi = model_sample_diff(model,
                                                    device=device,
                                                    dataloader=dataloader,
                                                    total_sample=x_t,
                                                    time=time,
                                                    is_condi=is_condi,
                                                    condi_flag=False,
                                                    vae=vae)
                model_output = (1 + omega) * model_output - omega * model_output_uncondi

      
        x_t, _ = noise_scheduler.step(model_output,  
                                     torch.from_numpy(np.array(time)).long().to(device),
                                      x_t,
                                      model_pred_type=model_pred_type)
      
        epoch_pcc = calculate_pcc_per_gene(x_t, gt)
        epoch_rmse = calculate_rmse_per_gene(x_t, gt)
        ts.set_postfix_str(f'PCC:{epoch_pcc:.5f}, RMSE:'
                           f'{epoch_rmse:.5f}')
        if mask is not None:
            x_t = x_t *  mask + (1 - mask) * gt

        if time == 0 and model_pred_type == 'x_start':
           
            sample = model_output


    recon_x = x_t.detach().cpu().numpy()
    return recon_x

