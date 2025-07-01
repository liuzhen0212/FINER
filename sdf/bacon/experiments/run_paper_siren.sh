## conda activate bacon
## bash run.sh

## SIREN
python train_sdf.py --config ./config/sdf/siren_armadillo.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type siren \
    --w0 30 \
    --experiment_name armadillo_siren_2x256 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/siren_dragon.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type siren \
    --w0 30 \
    --experiment_name dragon_siren_2x256 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/siren_lucy.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type siren \
    --w0 30 \
    --experiment_name lucy_siren_2x256 \
    --gpu 0 # &

python train_sdf.py --config ./config/sdf/siren_thai.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type siren \
    --w0 30 \
    --experiment_name thai_siren_2x256 \
    --gpu 0 # &

# wait