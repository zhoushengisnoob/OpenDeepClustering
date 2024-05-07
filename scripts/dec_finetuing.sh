# mnist
# please be sure to let class num equal to the last element of dims
nohup python -u models/Simultaneous/DEC/main.py \
    --reuse True \
    --dataset_name MNIST \
    --dataset_dir ~/dataset \
    --class_num 10 \
    --grey True \
    --img_size_at 28 28 \
    --optimizer sgd \
    --lr 0.01 \
    --weight_decay 0 \
    --sgd_momentum 0.9 \
    --use_vision False \
    --epochs 200 \
    --batch_size 512 \
    --eval_batch_size 512 \
    --num_workers 16 \
    --verbose True \
    --eval_step 20 \
    --save_step 20 \
    --dims 500 500 2000 10 \
    >./exps/mnist/dec/finetuing.log &
