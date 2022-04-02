import pandas as pd
import argparse
import sys
from util.util import Models, get_args

pd.options.display.max_columns = None
pd.options.display.max_rows = None


def main(m, t, e):
    from train.train import Train
    from test.test import Evaluate
    for model in m.current_models:
        if t and e:
            trained = Train(8, 175, model)
            Evaluate(model, trained)
        elif t:
            Train(8, 175, model)
        elif e:
            Evaluate(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build malicious URL detection models and evaluate them.')
    parser.add_argument('-m', '--models', choices=get_args(), help="Specify which models to train / evaluate.", required=True)
    parser.add_argument('-t', '--train', default=True, action='store_true', help="Specify whether to train models or not.")
    parser.add_argument('-e', '--evaluate', default=True, action='store_true', help="Specify whether to evaluate models or not.")
    args = parser.parse_args(sys.argv[1:])
    main(Models(vars(args)['models']), vars(args)['train'], vars(args)['evaluate'])
