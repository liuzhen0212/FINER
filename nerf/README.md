# FINER: Flexible spectral-bias tuning in Implicit NEural Representation by Variable-periodic Activation Functions

To better verify the advantages of FINER for representing high-frequency components, we follow the experimental setting from [WIRE](https://arxiv.org/pdf/2301.05187) (Sec. 4.3), where only 25 images are used for training, and each image is downsampled to a resolution of $200\times200$.

Thanks to [torch-ngp](https://github.com/ashawkey/torch-ngp), we made only minor modifications. Specifically, we removed positional encoding and replace the ReLU-MLP with FINER, following the [WIRE code](https://github.com/vishwa91/wire/files/12441797/network.txt).

Finally, please add the **[network_finer.py](network_finer.py)** file to the [torch-ngp/nerf](https://github.com/ashawkey/torch-ngp/tree/main/nerf) directory, and replace [torch-ngp/nerf/provider.py](https://github.com/ashawkey/torch-ngp/blob/main/nerf/provider.py) and [torch-ngp/main_nerf.py](https://github.com/ashawkey/torch-ngp/blob/main/main_nerf.py) with the modified **[provider.py](provider.py)** and **[main_nerf.py](main_nerf.py)** files, respectively. 

```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py ../data/nerf_synthetic/drums \
    --nn finer --lr 2e-4 --iter 37500 --downscale 4 \
    --trainskip 4 \
    --num_layers 4 --hidden_dim 182 --geo_feat_dim 182 --num_layers_color 4 --hidden_dim_color 182 \
    --workspace logs/drums_finer \
    -O --bound 1 --scale 0.8 --dt_gamma 0
```