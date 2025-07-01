## conda activate bacon
## bash run.sh

FBS_TUPLE=(1)
GPU_TUPLE=(2)
LR_TUPLE=(1e-4)
DATA_TUPLE=("armadillo" "dragon" "lucy" "thai")

# DATA="armadillo"
for ((k=0; k<${#DATA_TUPLE[@]}; k++)); do
    DATA=${DATA_TUPLE[k]}

    for ((i=0; i<${#FBS_TUPLE[@]}; i++)); do
        FBS=${FBS_TUPLE[i]}
        GPU=${GPU_TUPLE[i]}
        LR=${LR_TUPLE[i]}

        echo "FBS:" $FBS "GPU:" $GPU "LR:" $LR "exp:" gf_${DATA}_fbs_$FBS

        python train_sdf.py --config ./config/sdf/wire_${DATA}.ini \
            --logging_root ../logs_Finer++/${DATA} \
            --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type gf \
            --fbs $FBS \
            --experiment_name ${DATA}_gf_fbs_$FBS \
            --gpu $GPU # &

        # if [ $(((i + 1) % 2)) -eq 0 ]; then
        #     # echo $i
        #     wait
        # fi
    done 
done 

