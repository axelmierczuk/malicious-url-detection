import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from util.util import TYPE, get_save_loc, Models


class Lexical:
    def __init__(self, data, label="label"):
        self.data = data
        self.main_label = label
        self.save_name = "lexical-forest"
        self.model = None
        self.models = {}
        self.predictions = None

    def load_model(self):
        self.model = joblib.load(get_save_loc(Models.lexicographical) + self.save_name + ".joblib")

    def export_model(self):
        if self.model is None:
            return
        joblib.dump(self.model, get_save_loc(Models.lexicographical) + self.save_name + ".joblib")

    def train_model(self):
        print("Training model.")
        self.model = self.build_model()
        self.model = self.fit()
        self.predictions, y = self.predict_discrete(TYPE.test)
        self.stats(TYPE.test, y)
        # self.plot()
        self.export_model()

    def build_model(self):
        print("Building pipeline.")
        return RandomForestClassifier(max_depth=10000, random_state=0)

    def fit(self):
        print("Fitting model.")

        y = self.data[TYPE.train].pop('label').to_numpy()
        x = self.data[TYPE.train].to_numpy()

        return self.model.fit(x, y)

    def predict_discrete(self, t):
        y = self.data[t].pop('label').to_numpy()
        x = self.data[t].to_numpy()
        return self.model.predict(x), y

    def predict_proba(self, elems):
        return self.model.predict_proba(elems)

    def stats(self, t, y):
        print(np.mean(self.predictions == y))

    def plot(self):
        if self.model is None:
            return
        if self.data is None:
            return
