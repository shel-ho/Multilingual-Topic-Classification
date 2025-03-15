from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from operator import itemgetter
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
from abc import ABC, abstractmethod

# Top 5 most spoken languages and one lang from each United Nations geoscheme region
# excluding the Americas because none in this dataset is supported by XLM-R
# and Viet for fun and round up to 10
# - Oceania: Samoan
# - Asia 1 (W & C): Arabic
# - Asia 2 (S): Hindi, Bengali
# - Asia 3 (SE & E): Chinese (Traditional), Vietnamese
# - Europe 1 (N, W, S): English, Spanish
# - Europe 2 (E): Slovak
# - Africa: Afrikaans
langs = ['smo_Latn', 'afr_Latn', 'zho_Hant', 'arb_Arab', 'hin_Deva',
         'eng_Latn', 'slk_Latn', 'spa_Latn', 'ben_Beng', 'vie_Latn']
num_langs = len(langs)
#Constants
train_size = 701
dev_size = 99
test_size = 204

#load dataset
def load_data():
    print("Loading Data")
    dsets = [load_dataset('Davlan/sib200', lang) for lang in langs]

    # Concatenate the datasets by split
    train_data = concatenate_datasets([dset['train'] for dset in dsets])
    dev_data = concatenate_datasets([dset['validation'] for dset in dsets])
    test_data = concatenate_datasets([dset['test'] for dset in dsets])

# Combine split datasets back into one
    data = DatasetDict(dict(train=train_data,validation=dev_data,test=test_data))
    print("Data loaded")
    return data

class FeatureExtractor(ABC):
    def __init__(self):
        self.feature_dicts = []
        self.train = None
        self.dev = None
        self.test = None

    @abstractmethod
    def extract_features(self, data):
        pass

    def vectorize(self):
        train_dev_split = num_langs*train_size
        dev_test_split = train_dev_split + num_langs*dev_size
        vectorizer = DictVectorizer(sparse=True)
        self.train = vectorizer.fit_transform(self.feature_dicts[:train_dev_split])
        self.dev = vectorizer.transform(self.feature_dicts[train_dev_split:dev_test_split])
        self.test = vectorizer.transform(self.feature_dicts[dev_test_split:])


class CharTrigramFeatExtractor(FeatureExtractor):
    def extract_features(self, data):
        for dset in data.values():
            for data_point in dset:
                text = data_point["text"]
                text = re.sub(r'[^\w\s]', '', text)
                self.feature_dicts.append({c1+c2+c3: 1.0 for c1, c2, c3 in zip(text, text[1:], text[2:])})


class TokenFeatureExtractor(FeatureExtractor):
    def extract_features(self, data):
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        for dset in data.values():
            for data_point in dset:
                text = data_point["text"]
                tokens = tokenizer.tokenize(text)
                self.feature_dicts.append({token: 1.0 for token in tokens})

class Model(ABC):
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.X_train = None
        self.y_train = None
        self.X_dev = None
        self.y_dev = None
        self.X_test = None
        self.y_test = None

    def get_labels(self):
        self.y_train = self.data["train"]["category"]
        self.y_dev = self.data["validation"]["category"]
        self.y_test = self.data["test"]["category"]

    def get_feat_vecs(self, extractor):
        extractor.extract_features(self.data)
        extractor.vectorize()

        self.X_train = extractor.train
        self.X_dev = extractor.dev
        self.X_test = extractor.test

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    @abstractmethod
    def grid_search(self):
        pass

    @abstractmethod
    def test(self):
        pass


class LogReg(Model):
    def grid_search(self):
        scores = defaultdict(float)
        C = [0.5, 1.0, 2.0]
        for c in C:
            lr = LogisticRegression(C=c)
            lr.fit(self.X_train, self.y_train)

            y_pred = lr.predict(self.X_dev)
            print(accuracy_score(self.y_dev, y_pred))

            test_y_pred = lr.predict(self.X_test)
            print(accuracy_score(self.y_test, test_y_pred))
            scores[repr(c)] = accuracy_score(self.y_dev, y_pred)

        print(max_item(scores))

    def test(self):
        y_pred = self.model.predict(self.X_test)

        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))

#Take out
class RandForest(Model):
    def grid_search(self):
        scores = defaultdict(float)

        bootstrap = [True, False]
        max_depth = [10 * i for i in range(1, 11)]
        max_depth.append(None)
        max_features = ['sqrt', 'log2']
        min_samples_leaf = [1, 2, 4]
        min_samples_split = [2, 5, 10]
        n_estimators = [200 * i for i in range(1, 11)]

        for strap in bootstrap:
            for depth in max_depth:
                for feature in max_features:
                    for min_leaf in min_samples_leaf:
                        for min_split in min_samples_split:
                            for estimator in n_estimators:
                                #print("Training " + repr((strap, depth, feature, min_leaf, min_split, estimator)))
                                forest = RandomForestClassifier(bootstrap=strap,
                                                                max_depth=depth,
                                                                max_features=feature,
                                                                min_samples_leaf=min_leaf,
                                                                min_samples_split=min_split,
                                                                n_estimators=estimator)

                                forest.fit(self.X_train, self.y_train)

                                y_pred = forest.predict(self.X_dev)
                                #print(accuracy_score(self.y_dev, y_pred))
                                scores[repr((strap, depth, feature,
                                             min_leaf, min_split, estimator))] = accuracy_score(self.y_dev, y_pred)

        print(max_item(scores))

    def test(self):
        y_pred = self.model.predict(self.X_test)

        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))

    def mini_sweep(self):
        max_features = ['sqrt', 'log2']
        n_estimators = [1000 + 200*i for i in range(0,6)]

        for feature in max_features:
            for estimator in n_estimators:
                print("Training " + repr((feature, estimator)))
                forest = RandomForestClassifier(bootstrap=False,
                                                max_depth=None,
                                                max_features=feature,
                                                min_samples_leaf=1,
                                                min_samples_split=2,
                                                n_estimators=estimator)

                forest.fit(self.X_train, self.y_train)

                y_pred = forest.predict(self.X_dev)
                print(accuracy_score(self.y_dev, y_pred))

                test_y_pred = forest.predict(self.X_test)
                print(accuracy_score(self.y_test, test_y_pred))


def max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    # import heapq
    # print(heapq.nlargest(5, scores, key=scores.get))
    return max(scores.items(), key=itemgetter(1))


def main():
    #load data
    data = load_data()

    #sweeping hyperparameters for each model
    trigrams = CharTrigramFeatExtractor
    tokens = TokenFeatureExtractor

    test_lr = LogReg(LogisticRegression(), data)
    extractors = [trigrams, tokens]
    sweep = False
    if sweep:
        for extractor in extractors:
            test_lr.get_labels()
            test_lr.get_feat_vecs(extractor())

            test_lr.grid_search()

    #Running each model with hyperparameters found in the sweep
    lr = LogReg(LogisticRegression(C=2.0), data)
    lr.get_labels()
    lr.get_feat_vecs(tokens())
    lr.train()
    lr.test()


if __name__ == "__main__":
    main()