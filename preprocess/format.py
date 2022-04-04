import random
import numpy as np
import pandas
import pandas as pd
import tensorflow as tf
import tld
from tqdm import tqdm
from tldextract import tldextract
from urllib.parse import urlparse
from util.util import get_characters, TYPE, Models, shannon, murmur


class PProcess:
    def __init__(self, batch_size, model_name, tensor_root=175):
        self.model_name = model_name
        self.tensor_length = int(tensor_root * tensor_root / 7)
        self.tensor_root = tensor_root
        self.char_len = 7
        self.main_label = 'label'
        self.seed = 42
        self.split_train = 0.8
        self.split_val = 0.1
        self.batch_size = batch_size
        self.test_labels = np.array([])
        self.data = None

    def buildmatrix_raw(self, url):
        char_obj = get_characters()
        main = np.zeros(shape=(self.tensor_length, self.char_len), dtype=np.byte)
        for cc, char in enumerate(url):
            if cc < self.tensor_length:
                main[cc] = np.array(char_obj.get(ord(char), [0, 0, 0, 0, 0, 0, 0]), dtype=np.byte)
        return np.reshape(main, (self.tensor_root, self.tensor_root))

    def buildmatrix_lexical(self, url, label):
        main = np.zeros(shape=(self.tensor_root + 1), dtype=np.float32)
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url
        try:
            parsed = urlparse(url)
        except:
            return None
        toplevel, domain, subdomain = tld.parse_tld(url)
        if parsed.netloc is None or toplevel is None:
            return None
        main[0] = 1 if tldextract.extract(parsed.netloc).subdomain != "" else 0
        main[1] = murmur(toplevel)
        main[2] = len(parsed.netloc[:len(toplevel) * -1])
        main[3] = parsed.port if parsed.port else 0
        main[4] = sum(c.isdigit() for c in url)
        main[5] = sum(c.isalpha() for c in url)
        main[6] = shannon(url)
        main[7] = int(label)
        return main

    def generator_lexical(self, t):
        dataset = self.data[t]
        main = np.zeros(shape=(self.batch_size, self.tensor_root), dtype=np.float32)
        labels = np.zeros(shape=(self.batch_size, 1), dtype=np.byte)
        for i in range(self.batch_size):
            v = random.randint(0, len(dataset.index) - 1)
            r = dataset.iloc[v]
            labels[i] = r[self.main_label]
            main[i] = np.array(r.values.tolist()[:-1], dtype=np.float32)
        return main, tf.keras.utils.to_categorical(labels, num_classes=2)

    def generator_raw(self, t):
        dataset = self.data[t]
        main = np.zeros(shape=(self.batch_size, self.tensor_root, self.tensor_root), dtype=np.byte)
        labels = np.zeros(shape=(self.batch_size, 1), dtype=np.byte)
        for i in range(self.batch_size):
            v = random.randint(0, len(dataset.index) - 1)
            url = dataset['url'][v]
            labels[i] = dataset['label'][v]
            main[i] = self.buildmatrix_raw(url)
        return main, tf.keras.utils.to_categorical(labels, num_classes=2)

    def generator(self, t, m):
        while True:
            if m == Models.raw:
                yield self.generator_raw(t)
            elif m == Models.lexicographical:
                yield self.generator_lexical(t)

    def build_df_processed_lexical(self, df):
        df_malicious = df.loc[df[self.main_label] == 1].reset_index(drop=True)
        df_benign = df.loc[df[self.main_label] == 0].reset_index(drop=True).head(len(df_malicious.index))
        df_malicious = df_malicious.append(df_benign)
        fin_arr = []
        fin_urls = []
        pbar = tqdm(total=len(df_malicious.index))
        for index, row in tqdm(df_malicious.iterrows()):
            pbar.update(1)
            tmp = self.buildmatrix_lexical(row['url'], row['label'])
            if tmp is not None:
                fin_arr.append(tmp.tolist())
                fin_urls.append(row['url'])
        df_final = pandas.DataFrame(fin_arr, columns=['subdomain', 'tld', 'len', 'port', 'digits', 'characters', 'entropy', 'label'])
        df_final.insert(8, "url", fin_urls, True)
        df_final.to_csv('data/lexical_dataset.csv', index=False)

    def build_df(self, s, replace_vals, label):
        """
        Used to format a dataframe such that it is compatible with the model, and other dataframes.
        """
        df = pd.read_csv(s)
        df.drop(df.columns.difference(['url', label]), 1, inplace=True)
        for k, v in replace_vals.items():
            df[k] = df[k].replace(v['id'], v['val'])
        if label != self.main_label:
            df = df.rename(columns={label: self.main_label})
        print(f"Built {s.split('/')[-1]} dataframe. ")
        return df

    def preprocess(self):
        """
        Main pre-processing function. Used to generate two Dataframes, one for testing and the other for training.
        """
        # # Used to generate main dataset
        # # Dataset from https://www.sciencedirect.com/science/article/pii/S2352340920311987?via%3Dihub#ecom0001
        # test = self.build_df('data/Webpages_Classification_test_data.csv',
        #                      {'label': {'id': ['good', 'bad'], 'val': [0, 1]}}, 'label')
        # train = self.build_df('data/Webpages_Classification_train_data.csv',
        #                       {'label': {'id': ['good', 'bad'], 'val': [0, 1]}}, 'label')
        #
        # # Dataset from https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
        # df = self.build_df('data/malicious_phish.csv',
        #                    {'type': {'id': ['benign', 'phishing', 'defacement', 'malware'], 'val': [0, 1, 1, 1]}},
        #                    'type')
        # df = df.append(test).append(train)
        #
        # self.build_df_processed_lexical(df)

        pd.options.display.max_columns = None
        pd.options.display.max_rows = None

        df = pd.read_csv('data/lexical_dataset.csv').sample(frac=1, random_state=self.seed).reset_index(drop=True)

        if self.model_name == "raw":
            df.drop(df.columns.difference(['url', 'label']), 1, inplace=True)
        elif self.model_name == "lexical":
            df = df.drop(['url'], axis=1)

        print("Original length: ", len(df))

        train_dataset, val_dataset, test_dataset = np.split(df.sample(frac=1, random_state=self.seed), [int(self.split_train * len(df)), int(1 - self.split_val * len(df))])

        train_dataset = train_dataset.reset_index(drop=True)
        val_dataset = val_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)

        # Sanity checks
        train_len = len(train_dataset.index)
        val_len = len(val_dataset.index)
        test_len = len(test_dataset.index)
        tot_len = train_len + val_len + test_len

        print("Train length: ", train_len)
        print("Validation length: ", val_len)
        print("Test length: ", test_len)
        print(f"Ratio: {int(train_len / tot_len * 100)}/{int(val_len / tot_len * 100)}/{int(test_len / tot_len * 100)}")

        self.data = {
            TYPE.train: train_dataset,
            TYPE.validation: val_dataset,
            TYPE.test: test_dataset
        }
