import joblib
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from util.util import TYPE, get_save_loc, Models


class NGram:
    def __init__(self, data, label="label", n=1):
        self.data = data
        self.main_label = label
        self.save_name = "pipeline-" + str(n)
        self.ngram = n
        self.model = None
        self.models = {}
        self.predictions = None

    def set_ngram(self, n):
        self.ngram = n
        self.save_name = "pipeline-" + str(n)

    def load_model(self):
        self.model = joblib.load(get_save_loc(Models.ngram) + self.save_name + ".joblib")

    def load_models(self):
        for i in range(1, 4):
            self.models[i] = joblib.load(get_save_loc(Models.ngram) + "pipeline-" + str(i) + ".joblib")

    def export_model(self):
        if self.model is None:
            return
        joblib.dump(self.model, get_save_loc(Models.ngram) + self.save_name + ".joblib")

    def train_model(self):
        print("Training model.")
        self.model = self.build_pipeline()
        self.model = self.fit()
        self.predictions = self.predict_discrete(TYPE.test)
        self.stats(TYPE.test)
        # self.plot()
        self.export_model()

    def build_pipeline(self):
        print("Building pipeline.")
        return Pipeline([('vect', CountVectorizer(ngram_range=(self.ngram, self.ngram))), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

    def fit(self):
        print("Fitting model.")
        return self.model.fit(self.data[TYPE.train]['url'], self.data[TYPE.train][self.main_label])

    def predict_discrete(self, t):
        return self.model.predict(self.data[t]['url'])

    def predict_proba(self, elem, i):
        return self.models[i].predict_proba(elem)

    def stats(self, t):
        print(np.mean(self.predictions == self.data[t][self.main_label]))

    def plot(self):
        if self.model is None:
            return
        if self.data is None:
            return
