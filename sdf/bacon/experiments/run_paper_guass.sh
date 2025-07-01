## conda activate bacon
## bash run.sh

python train_sdf.py --config ./config/sdf/wire_armadillo.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type gauss \
    --experiment_name armadillo_gauss_2x256_s30 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/wire_dragon.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type gauss \
    --experiment_name dragon_gauss_2x256_s30 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/wire_thai.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type gauss \
    --experiment_name thai_gauss_2x256_s30 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/wire_lucy.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type gauss \
    --experiment_name lucy_gauss_2x256_s30 \
    --gpu 0 # &
    
# wait
