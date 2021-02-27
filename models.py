import random

from nltk.lm.preprocessing import padded_everygram_pipeline


class MyKFold:
    def __init__(self, n_fold: int = 5, shuffle: bool = False):
        self.__n_fold = n_fold
        self.__shuffle = shuffle

    @property
    def n_fold(self):
        return self.__n_fold

    @n_fold.setter
    def n_fold(self, value: int):
        if not value >= 2:
            raise ValueError()
        self.__n_fold = value

    @property
    def shuffle(self):
        return self.__shuffle

    @shuffle.setter
    def shuffle(self, value: bool):
        self.__shuffle = value

    def __call__(self, lists):
        new_lists = lists.copy()
        if self.shuffle:
            random.shuffle(new_lists)
        total = len(new_lists)  # total number
        each = total // self.n_fold  # each
        first = total % self.n_fold
        if first == 0:
            first = each
        slots = []
        i = 1
        bgn = 0
        end = first
        slots.append(new_lists[bgn:end])
        while i < self.n_fold:
            bgn = end
            end += each
            slots.append(new_lists[bgn:end])
            i += 1
        for i in range(0, self.n_fold):
            train = []
            test = []
            for j in range(0, self.n_fold):
                if i != j:
                    train.extend(slots[j])
                else:
                    test.extend(slots[j])
            yield train, test


def init_models(n: int, func) -> list:
    """
    Creating a series of models, given a model initializer function and number of models.
    :param n: Numbers of models.
    :param func: Initializer function.
    :return: An array of models.
    """
    return [func(i) for i in range(n)]


def padded_multiple_models(n: int, datasets_lists: list) -> list:
    """
    Creating ngram and vocabs.
    :param n: Numbers of models.
    :param datasets_lists:
    :return: [(data_ngram, ngram_vocab), ...]
    """
    ans = []
    for i in range(n):
        ngram, vocab = padded_everygram_pipeline(i + 1, datasets_lists)
        ans.append((ngram, vocab))
    return ans


def fit_multiple_models(models: list, ngrams: list):
    assert len(models) == len(ngrams)
    for i in range(len(models)):
        models[i].fit(*ngrams[i])


def compress(corpora) -> list:
    flatten = []
    for sentences in corpora:
        new_sentence = []
        for sentence in sentences:
            new_sentence.extend(sentence)
        flatten.append(new_sentence)
    return flatten
