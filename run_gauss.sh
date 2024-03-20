MODELTYPE=gauss
EXPNAME=Gauss
LR=0.005
SCALE=30

## single
GPUID=0
IMGID=0
echo "GPUID:" $GPUID  "ImageId:" $IMGID
# train
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py \
    --model_type $MODELTYPE --exp_name $EXPNAME \
    --scale $SCALE \
    --lr $LR \
    --img_id $IMGID 
    
