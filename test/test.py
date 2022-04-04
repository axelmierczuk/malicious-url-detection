import tensorflow as tf
import numpy as np
import json
from util.util import TYPE, get_save_loc
from preprocess.format import PProcess
from tqdm import tqdm


class Evaluate:
    def __init__(self, m, trained=None, batch_size=None, size=None):
        self.save_location = get_save_loc(m)
        self.model_name = m

        if trained is None:
            self.model = tf.keras.models.load_model(self.save_location + 'saved-model/')
            self.processed = PProcess(batch_size, self.model_name, size)
            self.processed.preprocess()
        else:
            self.model = trained.model
            self.processed = trained.processed

        self.HP = 0.85
        self.MP = 0.725
        self.LP = 0.65

        self.size = size
        self.test = self.processed.generator(TYPE.test, m)
        self.batch_size = self.processed.batch_size
        self.labels = np.array([])
        self.predictions = None
        self.statistics = None

    def stats(self):
        mal_index = np.nonzero(self.labels)[0]
        benign_index = np.argwhere(self.labels == [0])[0]
        res = {
            'total-results': len(self.labels),
            'total-malicious': len(mal_index),
            'false-positive': {
                'total': {
                    "comment": "The model claims that a URL is malicious, when it is not.",
                    'count': 0,
                    'percentage': 0.0
                },
                'high-probability': {
                    "comment": "The model is extremely sure that a URL is malicious, when it is not.",
                    'count': 0,
                    'percentage': 0.0
                },
                'medium-probability': {
                    "comment": "The model is fairly sure that a URL is malicious, when it is not.",
                    'count': 0,
                    'percentage': 0.0
                },
                'low-probability': {
                    "comment": "The model is somewhat sure that a URL is malicious, when it is not.",
                    'count': 0,
                    'percentage': 0.0
                },
            },
            'false-negative': {
                'total': {
                    "comment": "The model claims that a URL is benign, when it is not.",
                    'count': 0,
                    'percentage': 0.0
                },
                'high-probability': {
                    "comment": "This are worst case scenario. The model is extremely sure that a URL is benign, when it is not.",
                    'count': 0,
                    'percentage': 0.0
                },
                'medium-probability': {
                    "comment": "The model is fairly sure that a URL is benign, when it is not.",
                    'count': 0,
                    'percentage': 0.0
                },
                'low-probability': {
                    "comment": "The model is somewhat sure that a URL is benign, when it is not.",
                    'count': 0,
                    'percentage': 0.0
                },
            }
        }
        for i in benign_index:
            if self.predictions[i][1] > self.HP:
                res['false-positive']['high-probability']['count'] = res['false-positive']['high-probability']['count'] + 1
            elif self.predictions[i][1] > self.MP:
                res['false-positive']['medium-probability']['count'] = res['false-positive']['medium-probability']['count'] + 1
            elif self.predictions[i][1] > self.LP:
                res['false-positive']['low-probability']['count'] = res['false-positive']['low-probability']['count'] + 1
            if self.predictions[i][1] > 0.5:
                res['false-positive']['total']['count'] = res['false-positive']['total']['count'] + 1

        for i in mal_index:
            if self.predictions[i][0] > self.HP:
                res['false-negative']['high-probability']['count'] = res['false-negative']['high-probability']['count'] + 1
            elif self.predictions[i][0] > self.MP:
                res['false-negative']['medium-probability']['count'] = res['false-negative']['medium-probability']['count'] + 1
            elif self.predictions[i][0] > self.LP:
                res['false-negative']['low-probability']['count'] = res['false-negative']['low-probability']['count'] + 1
            if self.predictions[i][0] > 0.5:
                res['false-negative']['total']['count'] = res['false-negative']['total']['count'] + 1

        res['false-positive']['high-probability']['percentage'] = float("{0:.2f}".format(res['false-positive']['high-probability']['count'] / res['total-results']))
        res['false-positive']['medium-probability']['percentage'] = float("{0:.2f}".format(res['false-positive']['medium-probability']['count'] / res['total-results']))
        res['false-positive']['low-probability']['percentage'] = float("{0:.2f}".format(res['false-positive']['low-probability']['count'] / res['total-results']))
        res['false-positive']['total']['percentage'] = float("{0:.2f}".format(res['false-positive']['total']['count'] / res['total-results']))

        res['false-negative']['high-probability']['percentage'] = float("{0:.2f}".format(res['false-negative']['high-probability']['count'] / res['total-malicious']))
        res['false-negative']['medium-probability']['percentage'] = float("{0:.2f}".format(res['false-negative']['medium-probability']['count'] / res['total-malicious']))
        res['false-negative']['low-probability']['percentage'] = float("{0:.2f}".format(res['false-negative']['low-probability']['count'] / res['total-malicious']))
        res['false-negative']['total']['percentage'] = float("{0:.2f}".format(res['false-negative']['total']['count'] / res['total-malicious']))

        self.statistics = res
        with open(self.save_location + 'data.json', 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, ensure_ascii=False, indent=4)

    def evaluate(self):
        """
        Main function used to evaluate a saved model's accuracy. Loads a saved model and runs a dataframe which stores
        testing data. Current evaluations give a result of
        """
        test_generator = tf.data.Dataset.from_generator(
            generator=lambda: self.test,
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                [self.batch_size, self.size, self.size] if self.model_name == "raw" else [self.batch_size, self.size, 1],
                [None, 2]
            )
        )
        results = self.model.evaluate(x=test_generator, steps=len(self.processed.data[TYPE.test].index) // self.batch_size, batch_size=self.batch_size)

        res = np.array([])
        dataset = test_generator.enumerate()
        pbar = tqdm(total=len(self.processed.data[TYPE.test]) // self.batch_size)
        for b in dataset.as_numpy_iterator():
            if b[0] >= len(self.processed.data[TYPE.test]) // self.batch_size:
                break
            else:
                pbar.update(1)
                self.labels = np.append(self.labels, np.array([i[1] for i in b[1][1]]))
                res = np.append(res, self.model.predict_step(b[1]))
        res = res.reshape((len(self.processed.data[TYPE.test]) // self.batch_size) * self.batch_size, 2)
        print(res)
        print("test loss, test acc:", results)

        self.predictions = res
