MODELTYPE=siren
EXPNAME=SIREN
LR=0.0005
FIRSTOMEGA=30
HIDDENOMEGA=30

GPUID=0
IMGID=0

# train
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py \
    --model_type $MODELTYPE --exp_name $EXPNAME \
    --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA \
    --lr $LR \
    --img_id $IMGID 

