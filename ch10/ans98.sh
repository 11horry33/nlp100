# 前処理
SRC_DICT=../data/pretrained_model_enja/dict.ja.txt
TGT_DICT=../data/pretrained_model_enja/dict.en.txt

fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --destdir ../corpus/98-bin \
    --trainpref ../corpus/train98 \
    --validpref ../corpus/dev98 \
    --testpref ../corpus/test98 \
    --srcdict $SRC_DICT \
    --tgtdict $TGT_DICT \
    --workers `nproc`


# 学習
PRETRAINED_MODEL=../data/pretrained_model_enja/base.pretrain.pt

CUDA_VISIBLE_DEVICES=0  fairseq-train ../corpus/98-bin \
    --restore-file $PRETRAINED_MODEL \
    --arch transformer \
    --task translation \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --lr 3e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.2 --clip-norm 0.0 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '{0.9, 0.98}' \
    --max-tokens 4096 --max-epoch 5 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --log-format simple --fp16 --save-interval 1\
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
    --eval-bleu-detok space \
    --save-dir ../results/98/checkpoints/ | tee -a ../results/98/train.log


# 生成
mkdir -p ../results/98/generate

fairseq-generate ../corpus/98-bin \
        --path ../results/98/checkpoints/checkpoint_best.pt \
        --batch-size 10 \
        --beam 5  > ../results/98/generate/result.txt

grep "^H-" ../results/98/generate/result.txt | sort -V |cut -f3 > ../results/98/generate/pred.src


# 評価
sacrebleu /home/horiguchi/nlp100/corpus/test98.en -i /home/horiguchi/nlp100/results/98/generate/pred.src

# 結果
# {
#  "name": "BLEU",
#  "score": 28.9,
#  "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1",
#  "verbose_score": "56.9/35.6/23.5/15.8 (BP = 0.982 ratio = 0.983 hyp_len = 28932 ref_len = 29443)",
#  "nrefs": "1",
#  "case": "mixed",
#  "eff": "no",
#  "tok": "13a",
#  "smooth": "exp",
#  "version": "2.3.1"
# }
