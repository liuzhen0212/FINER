MODELTYPE=pemlp
EXPNAME=PEMLP
LR=0.001
NFREQS=10

GPUID=0
IMGID=0

# train
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py \
    --model_type $MODELTYPE --exp_name $EXPNAME \
    --N_freqs $NFREQS --not_zero_mean \
    --lr $LR \
    --img_id $IMGID 
