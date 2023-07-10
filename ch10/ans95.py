import sentencepiece as spm
import sys

args = sys.argv

def spm_train(lang):
    spm.SentencePieceTrainer.Train("--input=../corpus/train."+lang+" --model_prefix=../spm_model/spm."+lang+" --vocab_size=8000 --character_coverage=0.9995 --model_type=unigram")

def tokenize(raw, lang):
    model = spm.SentencePieceProcessor(model_file="../spm_model/spm."+lang+".model")
    fin = open("../corpus/"+raw+"."+lang, "r")
    f = open("../corpus/"+raw+"95."+lang, "w")
    for line in fin:
        src = model.encode(line.strip(), out_type=str)
        f.write(" ".join(src) + "\n")  
    fin.close()
    f.close()

if __name__ == '__main__':
    spm_train(args[1])
    tokenize("train", args[1])
    tokenize("dev", args[1])
    tokenize("test", args[1])
