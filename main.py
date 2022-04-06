from urllib.parse import urlparse

import pandas as pd
import argparse
import sys
import tensorflow as tf
import numpy as np
import csv

from tldextract import tldextract
from preprocess.format import PProcess
from train.train_ngram import NGram
from train.train_lexical import Lexical
from util.util import Models, get_args, get_save_loc, get_type, get_characters, shannon
from timeit import default_timer as timer

pd.options.display.max_columns = None
pd.options.display.max_rows = None


class API:
    def __init__(self, urls=[]):
        self.urls = urls

        model_name = Models.raw
        save_location = get_save_loc(model_name)

        self.model_raw = tf.keras.models.load_model(save_location + 'saved-model/')
        self.model_lexical = ""
        self.model_ngram = NGram([])
        self.model_ngram.load_models()
        self.model_lexical = Lexical([])
        self.model_lexical.load_model()

    def set_urls(self, urls):
        self.urls = urls

    def buildmatrix_raw(self, url):
        char_obj = get_characters()
        main = np.zeros(shape=(1, 4375, 7), dtype=np.byte)
        for cc, char in enumerate(url):
            if cc < 4375:
                main[0][cc] = np.array(char_obj.get(ord(char), [0, 0, 0, 0, 0, 0, 0]), dtype=np.byte)
        return np.reshape(main, (1, 175, 175))

    def buildmatrix_lexical(self, url):
        main = np.zeros(shape=17, dtype=np.float32)
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url
        try:
            parsed = urlparse(url)
        except:
            return None
        if parsed.netloc is None:
            return None
        try:
            p = int(parsed.port)
        except:
            p = None

        main[0] = 1 if tldextract.extract(parsed.netloc).subdomain != "" else 0
        main[1] = len(url)
        main[2] = p if p else 0
        main[3] = sum(c.isdigit() for c in url)
        main[4] = sum(c.isalpha() for c in url)
        main[5] = sum(c == "." for c in url)
        main[6] = sum(c == "-" for c in url)
        main[7] = 1 if "@" in url else 0
        main[8] = 1 if "~" in url else 0
        main[9] = sum(c == "_" for c in url)
        main[10] = sum(c == "%" for c in url)
        main[11] = sum(c == "&" for c in url)
        main[12] = sum(c == "#" for c in url)
        main[13] = len(parsed.path.split('/'))
        main[14] = 1 if "//" in parsed.path else 0
        main[15] = len(parsed.query)
        main[16] = shannon(url)
        return main

    def build_report(self, save_location, name):
        start = timer()
        raw_results = np.array([])
        for url in self.urls:
            raw_results = np.append(raw_results, self.model_raw.predict_step(self.buildmatrix_raw(url)))
        tmp = []
        for url in self.urls:
            tmp.append(self.buildmatrix_lexical(url))
        pd_df = {
            'raw_benign_scores': raw_results.reshape((len(self.urls)), 2)[:, 0],
            'raw_malicious_scores': raw_results.reshape((len(self.urls)), 2)[:, 1],
            'ngram_1_benign_scores': np.array(self.model_ngram.predict_proba(self.urls, 1))[:, 0],
            'ngram_1_malicious_scores': np.array(self.model_ngram.predict_proba(self.urls, 1))[:, 1],
            'ngram_2_benign_scores': np.array(self.model_ngram.predict_proba(self.urls, 2))[:, 0],
            'ngram_2_malicious_scores': np.array(self.model_ngram.predict_proba(self.urls, 2))[:, 1],
            'ngram_3_benign_scores': np.array(self.model_ngram.predict_proba(self.urls, 3))[:, 0],
            'ngram_3_malicious_scores': np.array(self.model_ngram.predict_proba(self.urls, 3))[:, 1],
            'lexical_benign_scores': np.array(self.model_lexical.predict_proba(tmp))[:, 0],
            'lexical_malicious_scores': np.array(self.model_lexical.predict_proba(tmp))[:, 1],
        }

        pd_df['final_benign_scores'] = np.add(np.add(np.add(np.add(pd_df['raw_benign_scores'], (pd_df['ngram_1_benign_scores'] / 3)), (pd_df['ngram_2_benign_scores'] / 3)), (pd_df['ngram_3_benign_scores'] / 3)), pd_df['lexical_benign_scores']) / 3
        pd_df['final_malicious_scores'] = np.add(np.add(np.add(np.add(pd_df['raw_malicious_scores'], (pd_df['ngram_1_malicious_scores'] / 3)), (pd_df['ngram_2_malicious_scores'] / 3)), (pd_df['ngram_3_malicious_scores'] / 3)), pd_df['lexical_malicious_scores']) / 3

        end = timer()
        with open(save_location + "/report-" + name + ".csv", 'w', newline='') as csvfile:
            fieldnames = ['num_items', 'execution_time (s)', 'execution_time_per_url (ms)', 'detection_accuracy', 'false_negative']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                'num_items': len(self.urls),
                'execution_time (s)': '{0:.2f}'.format(end - start),
                'execution_time_per_url (ms)': '{0:.2f}'.format((end - start) / len(self.urls) * 1000)
            })
            writer.writerow({})
        pd.DataFrame.from_dict(pd_df).to_csv(save_location + "/report-" + name + ".csv", index=False, mode='a')



    def make_predictions(self):
        start = timer()
        raw_results = np.array([])
        for url in self.urls:
            raw_results = np.append(raw_results, self.model_raw.predict_step(self.buildmatrix_raw(url)))
        res_raw = raw_results.reshape((len(self.urls)), 2)[:, 1]

        res_ngram_1 = np.array(self.model_ngram.predict_proba(self.urls, 1))[:, 1]
        res_ngram_2 = np.array(self.model_ngram.predict_proba(self.urls, 2))[:, 1]
        res_ngram_3 = np.array(self.model_ngram.predict_proba(self.urls, 3))[:, 1]
        tmp = []
        for url in self.urls:
            tmp.append(self.buildmatrix_lexical(url))
        res_lexical = np.array(self.model_lexical.predict_proba(tmp))[:, 1]

        final_arr = np.add(np.add(np.add(np.add(res_raw, (res_ngram_1 / 3)), (res_ngram_2 / 3)), (res_ngram_3 / 3)), res_lexical) / 3

        end = timer()
        print(f"Modeling executed in {end - start} seconds.")

        return res_raw, res_ngram_1, res_ngram_2, res_ngram_3, res_lexical, np.array(final_arr)


def main(m, t, e, b, ep):
    from train.train import Train
    from test.test import Evaluate

    for model in m.current_models:
        if t and e:
            if model == Models.ngram:
                processed = PProcess(0, Models.ngram, 0)
                processed.preprocess()
                ngram = NGram(processed.data)
                ngram.train_model()
                ngram.set_ngram(2)
                ngram.train_model()
                ngram.set_ngram(3)
                ngram.train_model()
            elif model == Models.lexicographical:
                processed = PProcess(0, Models.lexicographical, 0)
                processed.preprocess()
                lexical = Lexical(processed.data)
                lexical.train_model()
            else:
                if ep is not None:
                    train = Train(b, m.get_size(model), model, epochs=ep)
                else:
                    train = Train(b, m.get_size(model), model)
                train.train()
                evaluate = Evaluate(model, train)
                evaluate.evaluate()
                evaluate.stats()
        elif t:
            if ep is not None:
                train = Train(b, m.get_size(model), model, epochs=ep)
            else:
                train = Train(b, m.get_size(model), model)
            train.train()
        elif e:
            evaluate = Evaluate(model, batch_size=b, size=m.get_size(model))
            evaluate.evaluate()
            evaluate.stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build malicious URL detection models and evaluate them.')
    parser.add_argument('-m', '--models', choices=get_args(), help="Specify which models to train / evaluate.", required=True)
    parser.add_argument('-t', '--train', default=False, action='store_true', help="Specify whether to train models or not.")
    parser.add_argument('-e', '--evaluate', default=False, action='store_true', help="Specify whether to evaluate models or not.")
    parser.add_argument('-b', '--batch_size', help="Batch sized to be used for computing.", type=int, required=True)
    parser.add_argument('--epochs', help="Epochs to be used for computing.", type=int)
    args = vars(parser.parse_args(sys.argv[1:]))
    main(Models(args['models']), args['train'], args['evaluate'], args['batch_size'], args['epochs'])
