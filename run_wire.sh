MODELTYPE=wire
EXPNAME=WIRE
LR=0.005
FIRSTOMEGA=20
HIDDENOMEGA=20
SCALE=10


GPUID=0
IMGID=0

# train
CUDA_VISIBLE_DEVICES=$GPUID python train_image.py \
    --model_type $MODELTYPE --exp_name $EXPNAME \
    --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA --scale $SCALE \
    --lr $LR \
    --img_id $IMGID 




# EXP: WIRE_lr1e-3  PSNR Mean: 21.104   PSNR(U8) Mean: 21.258
# EXP: WIRE_lr2e-3  PSNR Mean: 26.354   PSNR(U8) Mean: 26.464
# EXP: WIRE_lr3e-3  PSNR Mean: 27.656   PSNR(U8) Mean: 27.741
# EXP: WIRE_lr4e-3  PSNR Mean: 28.220   PSNR(U8) Mean: 28.282
# EXP: WIRE_lr5e-3  PSNR Mean: 28.780   PSNR(U8) Mean: 28.826
# EXP: WIRE_lr6e-3  PSNR Mean: 28.593   PSNR(U8) Mean: 28.634
# EXP: WIRE_lr7e-3  PSNR Mean: 28.721   PSNR(U8) Mean: 28.756
# EXP: WIRE_lr8e-3  PSNR Mean: 28.579   PSNR(U8) Mean: 28.608
# EXP: WIRE_lr9e-3  PSNR Mean: 24.826   PSNR(U8) Mean: 24.835
# EXP: WIRE_lr1e-2  PSNR Mean: 24.002   PSNR(U8) Mean: 24.005
# EXP: WIRE_lr2e-2  PSNR Mean: 12.582   PSNR(U8) Mean: 12.581
# EXP: WIRE_lr3e-2  PSNR Mean: 12.149   PSNR(U8) Mean: 12.148
# EXP: WIRE_lr4e-2  PSNR Mean: 11.653   PSNR(U8) Mean: 11.653
# EXP: WIRE_lr5e-2  PSNR Mean: 11.639   PSNR(U8) Mean: 11.638
# EXP: WIRE_lr6e-2  PSNR Mean: 11.521   PSNR(U8) Mean: 11.521
# EXP: WIRE_lr7e-2  PSNR Mean: 11.520   PSNR(U8) Mean: 11.519
# EXP: WIRE_lr8e-2  PSNR Mean: 11.520   PSNR(U8) Mean: 11.520
# EXP: WIRE_lr9e-2  PSNR Mean: 11.520   PSNR(U8) Mean: 11.519