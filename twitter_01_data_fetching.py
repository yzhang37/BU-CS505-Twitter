from config import auth, PATH_STEP1_RAW
from twython import Twython
import ujson as json
import os
import time


twitter = Twython(
    app_key=auth.api_key, app_secret=auth.api_secret,
    oauth_token=auth.access_token, oauth_token_secret=auth.access_token_secret
)
# twitter.verify_credentials()


def fetch_twitter(count=10):
    results = twitter.search(
        q="covid",
        lang="en",
        result_type="recent",
        tweet_mode="extended",
        count=str(count)
    )
    result_texts = results["statuses"]
    return result_texts


def extract_text(status):
    if "retweeted_status" in status.keys():
        return status["retweeted_status"]["full_text"]
    else:
        return status["full_text"]


def continuous_save_twitter():
    raw_json_file = PATH_STEP1_RAW
    raw_json_file = os.path.abspath(os.path.join('.', raw_json_file))

    raw_texts = []
    if os.path.exists(raw_json_file) and os.path.isfile(raw_json_file):
        with open(raw_json_file, 'r') as fin:
            raw_texts = json.load(fin)
    count = len(raw_texts)
    if count > 0:
        print(f"Preloaded {count} texts.")

    while count < 10000:
        new_texts = fetch_twitter(100)
        raw_texts.extend(new_texts)
        count = len(raw_texts)
        print(f"Outputting {count} texts...")
        with open(raw_json_file, 'w') as fout:
            json.dump(raw_texts, fout, ensure_ascii=False)
        time.sleep(2)


if __name__ == "__main__":
    continuous_save_twitter()
