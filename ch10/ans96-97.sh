# 学習
# tensorboard --logdir ../results/96-97/log96-97 --host=localhostでログ確認
CUDA_VISIBLE_DEVICES=0  fairseq-train ../corpus/95-bin \
    --arch transformer \
    --task translation \
    --tensorboard-logdir ../results/96-97/log96-97 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --lr 3e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.2 --clip-norm 0.0 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '{0.9, 0.98}' \
    --max-tokens 4096 --max-epoch 30 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --log-format simple --fp16 --save-interval 2\
    --patience 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
    --eval-bleu-detok space \
    --save-dir ../results/96-97/checkpoints/ | tee -a ../results/96-97/train.log


# 生成
mkdir -p ../results/96-97/generate

fairseq-generate ../corpus/95-bin \
        --path ../results/96-97/checkpoints/checkpoint_best.pt \
        --batch-size 10 \
        --beam 5  > ../results/96-97/generate/result.txt

grep "^H-" ../results/96-97/generate/result.txt | sort -V |cut -f3 > ../results/96-97/generate/pred.src


# 評価
sacrebleu /home/horiguchi/nlp100/corpus/test95.en -i /home/horiguchi/nlp100/results/96-97/generate/pred.src

# 結果
# {
#  "name": "BLEU",
#  "score": 10.9,
#  "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1",
#  "verbose_score": "35.1/15.2/7.5/3.9 (BP = 0.982 ratio = 0.982 hyp_len = 28435 ref_len = 28945)",
#  "nrefs": "1",
#  "case": "mixed",
#  "eff": "no",
#  "tok": "13a",
#  "smooth": "exp",
#  "version": "2.3.1"
# }