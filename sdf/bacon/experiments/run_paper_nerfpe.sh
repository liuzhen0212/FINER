## conda activate bacon
## bash run.sh
# PE(NeRF) 

python train_sdf.py --config ./config/sdf/nerfpe_armadillo.ini \
    --lr 0.0005 --hidden_size 256 --hidden_layers 2 --model_type nerfpe \
    --experiment_name armadillo_nerfpe_2x256_L10 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/nerfpe_dragon.ini \
    --lr 0.0005 --hidden_size 256 --hidden_layers 2 --model_type nerfpe \
    --experiment_name dragon_nerfpe_2x256_L10 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/nerfpe_lucy.ini \
    --lr 0.0005 --hidden_size 256 --hidden_layers 2 --model_type nerfpe \
    --experiment_name lucy_nerfpe_2x256_L10 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/nerfpe_thai.ini \
    --lr 0.0005 --hidden_size 256 --hidden_layers 2 --model_type nerfpe \
    --experiment_name thai_nerfpe_2x256_L10 \
    --gpu 0 # &

# wait


