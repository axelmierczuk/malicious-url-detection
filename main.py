import pandas as pd
from train.train import Train
from test.test import Evaluate

pd.options.display.max_columns = None
pd.options.display.max_rows = None


def main():
    trained = Train(8, 175)
    evaluated = Evaluate(trained)


if __name__ == "__main__":
    main()
