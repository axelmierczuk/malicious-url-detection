import pandas as pd
import argparse
import sys

from preprocess.format import PProcess
from train.train_ngram import NGram
from util.util import Models, get_args, get_save_loc, get_type

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# TODO To make this work, will need to do if statements. It is the only way to achieve granularity for each model.
# def make_prediction(urls):
#     import tensorflow as tf
#     import numpy as np
#     import joblib
#
#     predictions = {}
#     for model_name in get_args():
#         t = get_type(model_name)
#         if t == Models.SKLEARN:
#             model = joblib.load(get_save_loc(Models.ngram) + model_name + ".joblib")
#             predictions[model_name] = np.array(model.predict_proba(urls))
#         elif t == Models.TF:
#             model = tf.keras.models.load_model(get_save_loc(model_name) + 'saved-model/')
#             predictions[model_name] = np.array(model.predict(urls))
#     print(predictions)


def main(m, t, e, b, ep):
    from train.train import Train
    from test.test import Evaluate

    for model in m.current_models:
        if t and e:
            if model == Models.ngram:
                processed = PProcess(0, "raw", 0)
                ngram = NGram(processed)
                ngram.train_model()
                ngram.set_ngram(2)
                ngram.train_model()
                ngram.set_ngram(3)
                ngram.train_model()
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
