# 生成
mkdir -p ../results/91/generate

fairseq-generate ../corpus/91-bin \
        --path ../results/91/checkpoints/checkpoint_best.pt \
        --batch-size 10 \
        --beam 5  > ../results/91/generate/result.txt

grep "^H-" ../results/91/generate/result.txt | sort -V |cut -f3 > ../results/91/generate/pred.src
