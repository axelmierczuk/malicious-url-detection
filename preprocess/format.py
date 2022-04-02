import random

import numpy as np
import pandas as pd
import tensorflow as tf

from util.util import get_characters, TYPE


class PProcess:
    def __init__(self, batch_size, tensor_root=175, is_loaded=False):
        self.tensor_length = int(tensor_root * tensor_root / 7)
        self.tensor_root = tensor_root
        self.char_len = 7
        self.main_label = 'label'
        self.seed = 42
        self.split_train = 0.8
        self.split_val = 0.1
        self.batch_size = batch_size
        self.parent_dir = "../" if is_loaded else ""
        self.test_labels = np.array([])
        self.data = self.preprocess()

    def buildmatrix(self, url):
        char_obj = get_characters()
        main = np.zeros(shape=(self.tensor_length, self.char_len), dtype=np.byte)
        for cc, char in enumerate(url):
            if cc < self.tensor_length:
                main[cc] = np.array(char_obj.get(ord(char), [0, 0, 0, 0, 0, 0, 0]), dtype=np.byte)
        return np.reshape(main, (self.tensor_root, self.tensor_root))

    def generator(self, t):
        dataset = self.data[t]
        while True:
            main = np.zeros(shape=(self.batch_size, self.tensor_root, self.tensor_root), dtype=np.byte)
            labels = np.zeros(shape=(self.batch_size, 1), dtype=np.byte)
            for i in range(self.batch_size):
                v = random.randint(0, len(dataset.index) - 1)
                url = dataset['url'][v]
                labels[i] = dataset['label'][v]
                main[i] = self.buildmatrix(url)
            if t == TYPE.test:
                self.test_labels = np.append(self.test_labels, labels)
            yield main, tf.keras.utils.to_categorical(labels, num_classes=2)

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
        print(f"Built {s.split('/')[-1]} dataframe: ")
        print(df.head())
        return df

    def preprocess(self):
        """
        Main pre-processing function. Used to generate two Dataframes, one for testing and the other for training.
        """
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None

        # Dataset from https://www.sciencedirect.com/science/article/pii/S2352340920311987?via%3Dihub#ecom0001
        test = self.build_df(self.parent_dir + 'data/Webpages_Classification_test_data.csv', {'label': {'id': ['good', 'bad'], 'val': [0, 1]}}, 'label')
        train = self.build_df(self.parent_dir + 'data/Webpages_Classification_train_data.csv', {'label': {'id': ['good', 'bad'], 'val': [0, 1]}}, 'label')

        # Dataset from https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
        df = self.build_df(self.parent_dir + 'data/malicious_phish.csv', {'type': {'id': ['benign', 'phishing', 'defacement', 'malware'], 'val': [0, 1, 1, 1]}}, 'type')

        df = df.append(test).append(train)

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

        return {
            TYPE.train: train_dataset,
            TYPE.validation: val_dataset,
            TYPE.test: test_dataset
        }
