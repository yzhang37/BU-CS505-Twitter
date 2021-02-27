from nltk.lm.models import MLE

from config import PATH_STEP2_CLEAN
from models import init_models, compress, padded_multiple_models, fit_multiple_models, MyKFold
from utilities import load_json, Ngrams


def process():
    data = load_json(PATH_STEP2_CLEAN)
    kFold = MyKFold(10, shuffle=True)
    print("MLE generating sentences using Language model.")
    for train_tweets, _ in kFold(data):
        models = init_models(3, MLE)
        train_sents = compress(train_tweets)

        # padded multiple models, returning a list of (ngram, vocab)
        ngrams = padded_multiple_models(3, train_sents)

        # train models using ngrams
        fit_multiple_models(models, ngrams)

        for i in range(3):
            print(f"\nGenerating {Ngrams(i)} sentences:")
            for j in range(10):  # generate 10 sentences for each ngram.
                new_sentence = []
                word = models[i].generate(text_seed=['<s>'])
                new_sentence.append(word)
                while word != '</s>':
                    word = models[i].generate(text_seed=[word])
                    new_sentence.append(word)
                print(f"#{j}: [{' '.join(new_sentence)}]")
        break


if __name__ == "__main__":
    process()
