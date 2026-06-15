import anndata as ad
import numpy as np
import pandas as pd
import sys
import pickle
import os
import datetime
import time as tm
from functools import partial
import scipy.stats as st
from scipy.stats import wasserstein_distance
import scipy.stats
import copy
from sklearn.model_selection import KFold
import pandas as pd
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import seaborn as sns
import torch
from causalfusion.models import model_dict
from model.diff_train import VAE
from scipy.spatial.distance import cdist
import h5py
import time
import sys
import pickle
import yaml
import argparse
from os.path import join
from IPython.display import display
from model.diff_model import DiT_diff
from model.diff_scheduler import NoiseScheduler
from model.diff_train import normal_train_diff
from model.sample import sample_diff
from preprocess.result_analysis import clustering_metrics
from preprocess.utils import *
from preprocess.data import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default='_sc.h5ad')
parser.add_argument("--st_data", type=str, default='_st.h5ad')
parser.add_argument("--document", type=str, default='dataset-HBC') 
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=1024)  # 2048
parser.add_argument("--hidden_size", type=int, default=1024)  # 512
parser.add_argument("--epoch", type=int, default=2000)
parser.add_argument("--diffusion_step", type=int, default=2000)
parser.add_argument("--learning_rate", type=float, default=3e-6)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--noise_std", type=float, default=10)
parser.add_argument("--pca_dim", type=int, default=100)
parser.add_argument("--head", type=int, default=64)
parser.add_argument("--mask_nonzero_ratio", type=float, default=0.2)
parser.add_argument("--mask_zero_ratio", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)
parser.add_argument("--eps", type=float, default=1e-15)
parser.add_argument("--max-ar-weight", type=float, default=2.0)
parser.add_argument("--min-ar-weight", type=float, default=1.0)
parser.add_argument("--ar-weight-schedule", type=str, default="linear")
parser.add_argument("--ar_step_decay", type=float, default=0.9)
parser.add_argument("--vae_lat", type=int, default=256)
parser.add_argument("--vae_sig", type=int, default=1)
parser.add_argument("--decoder_train", type=int, default=0)



args = parser.parse_args()

print(os.getcwd())
print(torch.cuda.get_device_name(torch.cuda.current_device()))




def train_valid_test():
    seed_everything(args.seed)
    st_path = 'datasets/data/' + args.document + '/st/' + args.document + args.st_data
    sc_path = 'datasets/data/' + args.document + '/sc/' + args.document + args.sc_data

    directory = 'vae_pretrain_save/' + args.document + '_ckpt/' + args.document + '_scdiff'
   

    if not os.path.exists(directory):
        os.makedirs(directory)
   
    save_path = os.path.join(directory, args.document + '.pt')

    json_directory = f'json_files/{args.document}'
    json_file_path = f"{json_directory}/{args.document}_indices.json"


    dataset = ConditionalDiffusionDataset(sc_path, st_path)
    (train_dataset, train_gene_names), (valid_dataset, valid_gene_names), (
    test_dataset, test_gene_names) = split_dataset_with_gene_names(dataset, train_ratio=0.7, val_ratio=0.2,
                                                                   test_ratio=0.1, random_state=42, json_file=json_file_path)

   

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    cell_num = dataset.sc_data.shape[1]
    spot_num = dataset.st_data.shape[1]
    sc_gene_num = dataset.sc_data.shape[0]
    st_gene_num = dataset.st_data.shape[0]
  
    hidden_dim = args.hidden_size  
    latent_dim = args.vae_lat  
   
    vae = VAE(input_dim=spot_num, 
              hidden_dim=hidden_dim, 
              latent_dim=latent_dim,
              input_options=[spot_num,cell_num],
              Sigmoid=args.vae_sig).to(args.device)
    Data =  args.document
    vae.load_state_dict(torch.load('vae_pretrain_models/'+Data+'/'+'vae_model_final.pt', map_location=args.device), strict=False)
    model = model_dict["CausalFusion-L"](input_size=int(np.sqrt(latent_dim)),depth=args.depth)
    
    model.to(args.device)
    diffusion_step = args.diffusion_step
    vae.train()

    model.train()

    if not os.path.isfile(save_path):
        normal_train_diff(model,
                          dataloader=train_dataloader,
                          lr=args.learning_rate,
                          num_epoch=args.epoch,
                          diffusion_step=diffusion_step,
                          device=args.device,
                          pred_type='noise',
                          mask_nonzero_ratio=args.mask_nonzero_ratio,
                          mask_zero_ratio=args.mask_zero_ratio,
                          vae=vae,
                          ar_step_decay=args.ar_step_decay,
                          decoder_train=args.decoder_train)
        torch.save(model.state_dict(), save_path)
        vae_save_path = os.path.join(directory, args.document + '_vae.pt')
        torch.save(vae.state_dict(), vae_save_path)  
       
    else:
        model.load_state_dict(torch.load(save_path))
        vae_save_path = os.path.join(directory, args.document + '_vae.pt')
        vae.load_state_dict(torch.load(vae_save_path))
        print("Model loaded successfully!!")

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    model.eval()
                            

    with torch.no_grad():
       test_gt = torch.stack([data for data, t, _ in test_dataset])
       test_sc = torch.stack([t for data, t, _ in test_dataset])
     
       prediction = sample_diff(model,
                                device=args.device,
                                dataloader=test_dataloader,
                                noise_scheduler=noise_scheduler,
                                mask_nonzero_ratio=0.3,
                                mask_zero_ratio = 0,
                                gt=test_gt,
                                sc=test_sc,
                                num_step=diffusion_step,
                                sample_shape=(test_gt.shape[0], test_gt.shape[1]),
                                is_condi=True,
                                sample_intermediate=diffusion_step,
                                model_pred_type='x_start',
                                is_classifier_guidance=False,
                                omega=0.9,
                                vae=vae
                                )

    return prediction, test_gt, test_gene_names



Data =  args.document
outdir = 'vae_pretrain_results/' + Data +'/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

hyper_directory = 'vae_pretrain_save/'+Data+'_ckpt/'+Data+'_hyper/'
hyper_file = Data + '_hyperameters.yaml'
hyper_full_path = os.path.join(hyper_directory, hyper_file)
if not os.path.exists(hyper_directory):
    os.makedirs(hyper_directory)
args_dict = vars(args)
with open(hyper_full_path, 'w') as yaml_file:
    yaml.dump(args_dict, yaml_file)

prediction_result, ground_truth, test_gene_num = train_valid_test()


gene_name = test_gene_num
prediction_result = prediction_result.T
ground_truth = ground_truth.numpy().T
pred_result = pd.DataFrame(prediction_result, columns=[gene_name])
original = pd.DataFrame(ground_truth, columns=[gene_name])
pred_result.to_csv(outdir + '/our_prediction.csv', header=True, index=True)
original.to_csv(outdir + '/original.csv', header=True, index=True)


