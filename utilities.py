import pickle
import typing
from enum import Enum

import ujson as json


class Ngrams(Enum):
    Unigram = 0
    Bigram = 1
    Trigram = 2


def save_model(data: typing.Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_model(path: str):
    with open(path, 'rb') as f:
        dat = pickle.load(f)
    return dat


def save_json(data: typing.Any, path: str, *args, **kwargs):
    with open(path, "w") as f:
        json.dump(data, f, *args, **kwargs)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
