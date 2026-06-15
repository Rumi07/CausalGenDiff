import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from model.diff_train import VAE
from preprocess.utils import *
from preprocess.data import *
from torch.optim import Adam
import yaml

# Set up the argument parser
parser = argparse.ArgumentParser(description="Train VAE for diffusion model")
parser.add_argument("--sc_data", type=str, default='_sc.h5ad')
parser.add_argument("--st_data", type=str, default='_st.h5ad')
parser.add_argument("--document", type=str, default='dataset-HBC')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--vae_lat", type=int, default=256)
parser.add_argument("--vae_sig", type=int, default=1)
parser.add_argument("--pca_dim", type=int, default=100)
parser.add_argument("--seed", type=int, default=3407)

args = parser.parse_args()


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

json_directory = f'json_files/{args.document}'
json_file_path = f"{json_directory}/{args.document}_indices.json"

# Dataset paths
sc_path = 'datasets/data/' + args.document + '/sc/' + args.document + args.sc_data
st_path = 'datasets/data/' + args.document + '/st/' + args.document + args.st_data

# Load dataset
dataset = ConditionalDiffusionDataset(sc_path, st_path)
(train_dataset, train_gene_names), (valid_dataset, valid_gene_names), (test_dataset, test_gene_names) = split_dataset_with_gene_names(dataset, train_ratio=0.7, val_ratio=0.2,
                                                                   test_ratio=0.1, random_state=42, json_file=json_file_path)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
cell_num = dataset.sc_data.shape[1]
spot_num = dataset.st_data.shape[1]
sc_gene_num = dataset.sc_data.shape[0]
st_gene_num = dataset.st_data.shape[0]

vae = VAE(input_dim=spot_num, 
          hidden_dim=args.hidden_size, 
          latent_dim=args.vae_lat,
          input_options=[spot_num,cell_num],
          Sigmoid=args.vae_sig).to(args.device)

class VAE_Loss(nn.Module):
    def __init__(self):
        super(VAE_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()  

    def forward(self, recon_x, x, mu, logvar):
        
        recon_loss = self.mse_loss(recon_x, x)

        
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        
        return recon_loss + kl_divergence

optimizer = Adam(vae.parameters(), lr=args.learning_rate)


vae.train()
Data =  args.document
save_dir = os.path.join('vae_pretrain_models', Data)
os.makedirs(save_dir, exist_ok=True)

for epoch in range(args.epoch):
    epoch_loss = 0.0
    for i, (x, _, _) in enumerate(train_dataloader):
        x = x.float().to(args.device)
        
        
        optimizer.zero_grad()
      
        reconstructed, mu, logvar = vae(x)
        
       
        loss = VAE_Loss()(reconstructed, x, mu, logvar)
        
       
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{args.epoch}], Loss: {avg_epoch_loss:.4f}")
    
    
    if (epoch + 1) % 500 == 0:
        save_path = 'vae_pretrain_models/'+Data+'/'+f"vae_model_epoch_{epoch+1}.pt"
        torch.save(vae.state_dict(), save_path)
        print(f"VAE model saved to {save_path}")


final_model_path = 'vae_pretrain_models/'+Data+'/'+"vae_model_final.pt"
torch.save(vae.state_dict(), final_model_path)
print(f"VAE model training completed. Final model saved to {final_model_path}")



