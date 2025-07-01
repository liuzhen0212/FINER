## conda activate bacon
## bash run.sh

python train_sdf.py --config ./config/sdf/wire_armadillo.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type wire \
    --experiment_name armadillo_wire_2x256_w20s10 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/wire_dragon.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type wire \
    --experiment_name dragon_wire_2x256_w20s10 \
    --gpu 0 # 

python train_sdf.py --config ./config/sdf/wire_thai.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type wire \
    --experiment_name thai_wire_2x256_w20s10 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/wire_lucy.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type wire \
    --experiment_name lucy_wire_2x256_w20s10 \
    --gpu 0 # &

wait
