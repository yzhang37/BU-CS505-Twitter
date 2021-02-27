import numpy as np
from nltk.lm.models import KneserNeyInterpolated
from tqdm import tqdm

from config import PATH_STEP2_CLEAN
from models import init_models, compress, padded_multiple_models, fit_multiple_models, MyKFold
from utilities import load_json, Ngrams


def process():
    # 1. First load the data into memory
    data = load_json(PATH_STEP2_CLEAN)

    # 2. Making 10-Fold Cross Validation
    kFold = MyKFold(10, False)
    print("Starting 10-Fold CV training/test")
    means = np.zeros((10, 3))
    for idx, (train_tweets, test_tweets) in enumerate(kFold(data)):
        print(f"Fold {idx}:")
        models = init_models(3, KneserNeyInterpolated)
        train_sents = compress(train_tweets)
        test_sents = compress(test_tweets)

        # padded multiple models, returning a list of (ngram, vocab)
        ngrams = padded_multiple_models(3, train_sents)

        # train models using ngrams
        fit_multiple_models(models, ngrams)

        test_ngrams = padded_multiple_models(3, test_sents)
        for n in range(0, 3):
            temp = [(models[n]).perplexity(i)
                    for i in tqdm((test_ngrams[n])[0], desc=f"perplexity of {Ngrams(n)}", total=len(test_sents))]
            means[idx, n] = np.mean(temp)
        print(f"run {idx}, unigram: {means[idx, 0]}, bigram: {means[idx, 1]}, trigram: {means[idx, 2]}")

    final_means = np.mean(means, axis=0)
    print("Final mean of 10-Fold CV:")
    print(f"unigram: {final_means[0]}, bigram: {final_means[1]}, trigram: {final_means[2]}")


if __name__ == "__main__":
    process()
