# 前処理
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --destdir ../corpus/95-bin \
    --trainpref ../corpus/train95 \
    --validpref ../corpus/dev95 \
    --testpref ../corpus/test95 \
    --workers `nproc`


# 学習
CUDA_VISIBLE_DEVICES=0  fairseq-train ../corpus/95-bin \
    --arch transformer \
     --task translation \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr 3e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --clip-norm 0.0 \
    --optimizer adam --max-tokens 4096 --max-epoch 30 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --log-format simple --fp16 --save-interval 2\
    --patience 5 \
    --save-dir ../results/95/checkpoints/ | tee -a ../results/95/train.log


# 生成
mkdir -p ../results/95/generate

fairseq-generate ../corpus/95-bin \
        --path ../results/95/checkpoints/checkpoint_best.pt \
        --batch-size 10 \
        --beam 5  > ../results/95/generate/result.txt

grep "^H-" ../results/95/generate/result.txt | sort -V |cut -f3 > ../results/95/generate/pred.src


# 評価
sacrebleu /home/horiguchi/nlp100/corpus/test95.en -i /home/horiguchi/nlp100/results/95/generate/pred.src

# 結果
# {
#  "name": "BLEU",
#  "score": 9.1,
#  "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1",
#  "verbose_score": "31.8/12.8/5.8/2.9 (BP = 1.000 ratio = 1.011 hyp_len = 29252 ref_len = 28945)",
#  "nrefs": "1",
#  "case": "mixed",
#  "eff": "no",
#  "tok": "13a",
#  "smooth": "exp",
#  "version": "2.3.1"
# }
