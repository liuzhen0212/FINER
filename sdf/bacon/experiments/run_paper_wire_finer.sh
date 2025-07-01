## conda activate bacon
## bash run.sh

FBS_TUPLE=($(echo "scale=20; 1/sqrt(3)" | bc))
GPU_TUPLE=(0)
LR_TUPLE=(1e-4)

DATA_TUPLE=("armadillo" "dragon" "lucy" "thai")

for ((k=0; k<${#DATA_TUPLE[@]}; k++)); do
    DATA=${DATA_TUPLE[k]}

    for ((i=0; i<${#FBS_TUPLE[@]}; i++)); do
        FBS=${FBS_TUPLE[i]}
        GPU=${GPU_TUPLE[i]}
        LR=${LR_TUPLE[i]}

        echo "FBS:" $FBS "GPU:" $GPU "LR:" $LR "exp:" wf_${DATA}_fbs_$FBS

        python train_sdf.py --config ./config/sdf/wire_${DATA}.ini \
            --logging_root ../logs_Finer++/${DATA} \
            --lr 0.0001 --hidden_size 256 --hidden_layers 2 --model_type wf \
            --fbs $FBS \
            --experiment_name ${DATA}_wf_fbs_$FBS \
            --gpu $GPU # &

    done 
    # if [ $(((i + 1) % 2)) -eq 0 ]; then
    #     # echo $i
    #     wait
    # fi
done 

