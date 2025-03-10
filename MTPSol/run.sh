# export CUDA_VISIBLE_DEVICES=0,2,3

# train_epochs=1
# batch_size=24
# modelname='MTPSol'
# torchrun --nnodes 1 --nproc-per-node 3 run.py \
#   --is_training 1 \
#   --model $modelname \
#   --cosine \
#   --lradj 'type_fix' \
#   --learning_rate 0.000001 \
#   --weight_decay 0.00001 \
#   --num_workers 5 \
#   --batch_size $batch_size \
#   --num_epochs $train_epochs \
#   --use_multi_gpu \
#   --patience 10 \
#   --gradient_accumulation \
#   --gradient_accumulation_step 8 \
#   --checkpoints './best-checkpoints/checkpoint.pth' 

export CUDA_VISIBLE_DEVICES=2
train_epochs=400
batch_size=20

modelname='MTPSol'
python run.py \
  --is_training 0 \
  --model $modelname \
  --cosine \
  --lradj 'type_fix' \
  --learning_rate 0.0001 \
  --weight_decay 0.02 \
  --num_workers 5 \
  --vali_rate 0.9 \
  --warmup_steps 400 \
  --batch_size $batch_size \
  --num_epochs $train_epochs \
  --patience 20 \
  --checkpoints './best-checkpoints/checkpoint.pth' 