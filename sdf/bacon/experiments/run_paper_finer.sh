## conda activate bacon
## bash run.sh

## FINER
python train_sdf.py --config ./config/sdf/finer_armadillo.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type finer \
    --w0 30 --fbs 1.0 \
    --experiment_name armadillo_finer_2x256_w30_fbs_1.0 \
    --gpu 0 

python train_sdf.py --config ./config/sdf/finer_dragon.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type finer \
    --w0 30 --fbs 1.0 \
    --experiment_name dragon_finer_2x256_w30_fbs_1.0 \
    --gpu 0 

python train_sdf.py --config ./config/sdf/finer_lucy.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type finer \
    --w0 30 --fbs 1.0 \
    --experiment_name lucy_finer_2x256_w30_fbs_1.0 \
    --gpu 0 

python train_sdf.py --config ./config/sdf/finer_thai.ini \
    --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type finer \
    --w0 30 --fbs 1.0 \
    --experiment_name thai_finer_2x256_w30_fbs_1.0 \
    --gpu 0 

