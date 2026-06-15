python main.py \
    --document dataset-MG \
    --device cuda:0 \
    --batch_size 64 \
    --hidden_size 1024 \
    --epoch 3000 \
    --diffusion_step 1000 \
    --learning_rate 3e-4 \
    --depth 4 \
    --noise_std 10 \
    --pca_dim 100 \
    --head 32 \
    --mask_nonzero_ratio 0.3 \
    --mask_zero_ratio 0.4 \
    --vae_sig 1 \
    --decoder_train 0 \
    --ar_step_decay 0.8 > dataset_MG_run.log
   