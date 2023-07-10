import sentencepiece as spm
import sys

args = sys.argv

def tokenize(raw, lang):
    model = spm.SentencePieceProcessor(model_file="../spm_model/enja_spm_models/spm."+lang+".nopretok.model")
    fin = open("../corpus/"+raw+"."+lang, "r")
    f = open("../corpus/"+raw+"98."+lang, "w")
    for line in fin:
        src = model.encode(line.strip(), out_type=str)
        f.write(" ".join(src) + "\n")  
    fin.close()
    f.close()

if __name__ == '__main__':
    tokenize("train", args[1])
    tokenize("dev", args[1])
    tokenize("test", args[1])
