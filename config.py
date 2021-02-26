import json
import os


class MyAuth:
    def __init__(self):
        try:
            with open("key.json", 'r') as fkey:
                data = json.load(fkey)
                self.__API_KEY = data["api_key"]
                self.__API_SECRET = data["api_secret"]
                self.__ACCESS_TOKEN = data["access_token"]
                self.__ACCESS_TOKEN_SECRET = data["access_token_secret"]
                self.__BEAR = data["bear"]
        except FileNotFoundError:
            print("key.json not found!!!")
            raise

    @property
    def api_key(self):
        return self.__API_KEY

    @property
    def api_secret(self):
        return self.__API_SECRET

    @property
    def access_token(self):
        return self.__ACCESS_TOKEN

    @property
    def access_token_secret(self):
        return self.__ACCESS_TOKEN_SECRET


auth = MyAuth()
CURDIR = os.path.curdir
PATH_STEP1_RAW = os.path.abspath(os.path.join(CURDIR, "raw_tweets.json"))
PATH_STEP2_CLEAN = os.path.abspath(os.path.join(CURDIR, "clean_tweets.json"))
