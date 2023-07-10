# 前処理
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --destdir ../corpus/91-bin \
    --trainpref ../corpus/train \
    --validpref ../corpus/dev \
    --testpref ../corpus/test \
    --workers `nproc`


# 学習
CUDA_VISIBLE_DEVICES=0  fairseq-train ../corpus/91-bin \
    --arch transformer \
     --task translation \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr 3e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --clip-norm 0.0 \
    --optimizer adam --max-tokens 4096 --max-epoch 10 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --log-format simple --fp16 --save-interval 2\
    --save-dir ../results/91/checkpoints/ | tee -a ../results/91/train.log
