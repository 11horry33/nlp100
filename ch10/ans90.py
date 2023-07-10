# 英語データセットの作成
tokenizer = spacy.load("en_core_web_sm")
for raw, tok in [
    ("../data/kftt-data-1.0/data/orig/kyoto-train.en", "../corpus/train.en"),
    ("../data/kftt-data-1.0/data/orig/kyoto-dev.en", "../corpus/dev.en"),
    ("../data/kftt-data-1.0/data/orig/kyoto-test.en", "../corpus/test.en"),
    ]:
    make_dataset(raw, tok, tokenizer)


# 日本語データセットの作成
tokenizer = spacy.load("ja_core_news_sm")
for raw, tok in [
    ("../data/kftt-data-1.0/data/orig/kyoto-train.ja", "../corpus/train.ja"),
    ("../data/kftt-data-1.0/data/orig/kyoto-dev.ja", "../corpus/dev.ja"),
    ("../data/kftt-data-1.0/data/orig/kyoto-test.ja", "../corpus/test.ja"),
    ]:
    make_dataset(raw, tok, tokenizer)
