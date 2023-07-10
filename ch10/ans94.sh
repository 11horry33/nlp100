# 生成
for N in 1 20 40
do
    fairseq-generate ../corpus/91-bin \
        --path ../results/91/checkpoints/checkpoint_best.pt \
        --batch-size 10 \
        --beam $N > ../results/94/result_beam$N.txt

    grep "^H-" ../results/94/result_beam$N.txt | sort -V |cut -f3 > ../results/94/beam_$N.src
done

# 評価
for N in 1 20 40
do
    sacrebleu ../corpus/test.en -i ../results/94/beam_$N.src > ../results/94/beam_$N.score
done
