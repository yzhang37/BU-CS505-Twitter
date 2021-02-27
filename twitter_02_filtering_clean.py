from config import PATH_STEP1_RAW, PATH_STEP2_CLEAN
from nltk.tokenize import sent_tokenize, TweetTokenizer
from nltk.lm.preprocessing import pad_both_ends
import ujson as json
twTknzr = TweetTokenizer()


# 1. 用来提取文本的函数
def extract_text(status):
    if "retweeted_status" in status.keys():
        return status["retweeted_status"]["full_text"]
    else:
        return status["full_text"]


# 2. 每个 Tweets 分句子
def sentence_segment(sent):
    return sent_tokenize(sent)


# 3. 每个句子分词 + padding
def my_pad_both_ends(sentence):
    return list(pad_both_ends(sentence, n=2))


def word_tokenize_sentpad(sentsList):
    return list(map(my_pad_both_ends, map(twTknzr.tokenize, map(str.lower, sentsList))))


def process():
    with open(PATH_STEP1_RAW, "r") as fRaw:
        data = json.load(fRaw)

    data = list(map(extract_text, data))
    data = list(map(sentence_segment, data))
    data = list(map(word_tokenize_sentpad, data))

    with open(PATH_STEP2_CLEAN, "w") as fClean:
        json.dump(data, fClean, ensure_ascii=False)


if __name__ == "__main__":
    process()
