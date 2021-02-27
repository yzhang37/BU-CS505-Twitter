from config import PATH_STEP2_CLEAN
import ujson as json
from nltk.lm.models import KneserNeyInterpolated
from nltk.util import everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline
import numpy as np
import pickle


def compress(corpora):
    flatten = []
    for sentences in corpora:
        new_sentence = []
        for sentence in sentences:
            new_sentence.extend(sentence)
        flatten.append(new_sentence)
    return flatten


def save_model(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def process():
    with open(PATH_STEP2_CLEAN, "r") as f:
        data = json.load(f)
    # BUGBUG: this can also be modified to use Cross Validation
    train_tweets = data[:9000]
    test_tweets = data[9000:10000]

    model1 = KneserNeyInterpolated(1)
    model2 = KneserNeyInterpolated(2)
    model3 = KneserNeyInterpolated(3)

    train_sents = compress(train_tweets)
    test_sents = compress(test_tweets)

    train_unigram, unigram_vocab = padded_everygram_pipeline(1, train_sents)
    train_bigram, bigram_vocab = padded_everygram_pipeline(2, train_sents)
    train_trigram, trigram_vocab = padded_everygram_pipeline(3, train_sents)

    model1.fit(train_unigram, unigram_vocab)
    print("Saving model1...")
    save_model(model1, "model1.pickle")
    model2.fit(train_bigram, bigram_vocab)
    print("Saving model2...")
    save_model(model2, "model2.pickle")
    model3.fit(train_trigram, trigram_vocab)
    print("Saving model3...")
    save_model(model3, "model3.pickle")

    test_unigram, _ = padded_everygram_pipeline(1, test_sents)
    test_bigram, _ = padded_everygram_pipeline(2, test_sents)
    test_trigram, _ = padded_everygram_pipeline(3, test_sents)

    uni_mean = np.mean([model1.perplexity(i) for i in test_unigram])
    bi_mean = np.mean([model2.perplexity(i) for i in test_bigram])
    tri_mean = np.mean([model3.perplexity(i) for i in test_trigram])

    print(uni_mean)
    print(bi_mean)
    print(tri_mean)


if __name__ == "__main__":
    process()
