# mnist
# please be sure to let class num equal to the last element of dims
nohup python -u models/Simultaneous/IDEC/pretrain.py \
    --dataset_name MNIST \
    --dataset_dir ~/dataset \
    --class_num 10 \
    --grey True \
    --img_size_at 28 28 \
    --optimizer sgd \
    --lr 0.1 \
    --weight_decay 0 \
    --sgd_momentum 0.9 \
    --use_vision False \
    --batch_size 256 \
    --num_workers 16 \
    --verbose True \
    --save_step 5000 \
    --dims 500 500 2000 10 \
    >./exps/mnist/dec/pretrain.log &

# nohup python -u models/Simultaneous/IDEC/pretrain.py \
#     --dataset_name STL10 \
#     --dataset_dir ~/dataset \
#     --class_num 10 \
#     --grey False \
#     --img_size_at 28 28 \
#     --optimizer sgd \
#     --lr 0.1 \
#     --weight_decay 0 \
#     --sgd_momentum 0.9 \
#     --use_vision False \
#     --batch_size 256 \
#     --num_workers 16 \
#     --verbose True \
#     --save_step 5000 \
#     --dims 500 500 2000 10 \
#     >./exps/stl10/dec/pretrain.log &

# nohup python -u models/Simultaneous/IDEC/pretrain.py \
#     --dataset_name CIFAR10 \
#     --dataset_dir ~/dataset \
#     --class_num 10 \
#     --grey False \
#     --img_size_at 32 32 \
#     --optimizer sgd \
#     --lr 0.1 \
#     --weight_decay 0 \
#     --sgd_momentum 0.9 \
#     --use_vision False \
#     --batch_size 256 \
#     --num_workers 16 \
#     --verbose True \
#     --save_step 5000 \
#     --dims 500 500 2000 10 \
#     >./exps/cifar10/dec/pretrain.log &
