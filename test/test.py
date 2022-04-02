import tensorflow as tf
import numpy as np
import json
from util.util import TYPE


class Evaluate:
    def __init__(self, trained):
        self.model = trained.model
        self.test = trained.processed.generator(TYPE.test)
        self.processed = trained.processed
        self.batch_size = self.processed.batch_size

        self.labels = np.array([])

        self.HP = 0.85
        self.MP = 0.725
        self.LP = 0.65

        self.predictions = self.evaluate()
        self.stats = self.stats()

        with open('model/data.json', 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=4)

    def stats(self):
        mal_index = np.nonzero(self.labels)[0]
        benign_index = np.argwhere(self.labels == [0])[0]
        res = {
            'total-results': len(self.labels),
            'total-malicious': len(mal_index),
            'false-positive': {
                'total': {
                    "comment": "The model is claims that a URL is malicious, when it is not.",
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
                    "comment": "The model is claims that a URL is benign, when it is not.",
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

        return res

    def evaluate(self):
        """
        Main function used to evaluate a saved model's accuracy. Loads a saved model and runs a dataframe which stores
        testing data. Current evaluations give a result of
        """
        test_generator = tf.data.Dataset.from_generator(
            generator=lambda: self.test,
            output_types=(tf.float32, tf.float32),
            output_shapes=([self.batch_size, self.processed.tensor_root, self.processed.tensor_root], [None, 2])
        )
        results = self.model.evaluate(x=test_generator, steps=len(self.processed.data[TYPE.test]) // self.batch_size, batch_size=self.batch_size)

        res = np.array([])
        dataset = test_generator.enumerate()
        for b in dataset.as_numpy_iterator():
            if b[0] >= len(self.processed.data[TYPE.test]) // self.batch_size:
                break
            else:
                self.labels = np.append(self.labels, np.array([i[1] for i in b[1][1]]))
                res = np.append(res, self.model.predict_step(b[1]))
        res = res.reshape((len(self.processed.data[TYPE.test]) // self.batch_size) * self.batch_size, 2)
        print("test loss, test acc:", results)
        return res
