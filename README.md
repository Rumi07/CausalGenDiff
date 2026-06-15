# CausalGenDiff: Generative Causal Diffusion Bridges scRNA-seq and Spatial Transcriptomics

We present CausalGenDiff, a model that integrates diffusion and autoregressive processes to exploit these underlying causal dependencies. Our approach extends the Causal Attention Transformer originally designed for image generation to handle high-dimensional gene expression data, enabling the capture of gene regulatory mechanisms without relying on predefined relationships. We further incorporate VAE-based pretraining and fine-tuning strategies to enhance performance, supported by thorough ablation studies.


## Setup
Create a conda environment using the environment.yml file
```
 conda env create -f environment.yml
```
Please consider installing any package manually via pip or conda if there is dependency issue.
## Data
All the datasets used in this paper can be downloaded from url：https://zenodo.org/records/12792074

## Preprocess data

To preprocess the data, run the `data_preprocess.py` script located in the `preprocess` directory. Use the following command:

```
python preprocess/data_preprocess.py --input data/raw_data.csv --output data/processed_data.csv
```

## Running Experiments
To train the vae, use the following command:
```
python train_vae.py
or for MG data the following command could be used:
bash vae_train.sh
```
To train the CausalGenDiff , use the following command:

```
python main.py
or for MG data, the following command could be used:
bash mg.sh
```

To evaluate the model, run:

```
python evaluate.py

```
</pre></div>
