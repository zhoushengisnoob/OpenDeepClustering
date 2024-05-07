# mnist
# please be sure to let class num equal to the last element of dims
nohup python -u models/Simultaneous/IDEC/main.py \
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
    --batch_size 256 \
    --eval_batch_size 256 \
    --num_workers 16 \
    --verbose True \
    --eval_step 20 \
    --save_step 20 \
    --dims 500 500 2000 10 \
    >./exps/mnist/dec/finetuing.log &

# nohup python -u models/Simultaneous/IDEC/main.py \
#     --reuse True \
#     --dataset_name STL10 \
#     --dataset_dir ~/dataset \
#     --class_num 10 \
#     --grey False \
#     --img_size_at 28 28 \
#     --optimizer sgd \
#     --lr 0.01 \
#     --weight_decay 0 \
#     --sgd_momentum 0.9 \
#     --use_vision False \
#     --epochs 200 \
#     --batch_size 256 \
#     --eval_batch_size 256 \
#     --num_workers 16 \
#     --verbose True \
#     --eval_step 10 \
#     --save_step 20 \
#     --dims 500 500 2000 10 \
#     --pretrain_path ~/download/OpenDeepClustering/model_saves/STL10/DEC/pretrain/ckpt_105000.pt \
#     >./exps/stl10/dec/finetuing.log &

# nohup python -u models/Simultaneous/IDEC/main.py \
#     --reuse True \
#     --dataset_name CIFAR10 \
#     --dataset_dir ~/dataset \
#     --class_num 10 \
#     --grey False \
#     --img_size_at 28 28 \
#     --optimizer sgd \
#     --lr 0.01 \
#     --weight_decay 0 \
#     --sgd_momentum 0.9 \
#     --use_vision False \
#     --epochs 200 \
#     --batch_size 256 \
#     --eval_batch_size 256 \
#     --num_workers 16 \
#     --verbose True \
#     --eval_step 10 \
#     --save_step 20 \
#     --dims 500 500 2000 10 \
#     --pretrain_path ~/download/OpenDeepClustering/model_saves/CIFAR10/DEC/pretrain/ckpt_100000.pt \
#     >./exps/cifar10/dec/finetuing.log &
