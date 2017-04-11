import sys
import os
import pickle
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from gensim.models import KeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

from nltk import sent_tokenize
from nltk import pos_tag
from nltk import map_tag
from nltk import word_tokenize
from nltk.corpus import stopwords

# Load NLTK's English stop-words list
stop_words = set(stopwords.words('english'))


#
# embeddings vector representations
#

def tag_pos(x):
    sentences = sent_tokenize(x.decode("utf8"))
    sents = []
    for s in sentences:
        text = word_tokenize(s)
        pos_tagged = pos_tag(text)
        simplified_tags = [
            (word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tagged]
        sents.append(simplified_tags)
    return sents


def post_tag_documents(data_df):
    x_data = []
    y_data = []
    total = len(data_df['plot'].as_matrix().tolist())
    plots = data_df['plot'].as_matrix().tolist()
    genres = data_df.drop(['plot', 'title', 'plot_lang'], axis=1).as_matrix()
    for i in range(len(plots)):
        sents = tag_pos(plots[i])
        x_data.append(sents)
        y_data.append(genres[i])
        i += 1
        if i % 5000 == 0:
            print i, "/", total

    return x_data, y_data


def word2vec(x_data, pos_filter):

    print "Loading GoogleNews-vectors-negative300.bin"
    google_vecs = KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True, limit=200000)

    print "Considering only", pos_filter
    print "Averaging Word Embeddings..."
    x_data_embeddings = []
    total = len(x_data)
    processed = 0
    for tagged_plot in x_data:
        count = 0
        doc_vector = np.zeros(300)
        for sentence in tagged_plot:
            for tagged_word in sentence:
                if tagged_word[1] in pos_filter:
                    try:
                        doc_vector += google_vecs[tagged_word[0]]
                        count += 1
                    except KeyError:
                        continue

        doc_vector /= count
        if np.isnan(np.min(doc_vector)):
            continue

        x_data_embeddings.append(doc_vector)

        processed += 1
        if processed % 10000 == 0:
            print processed, "/", total

    return np.array(x_data_embeddings)


def doc2vec(data_df):
    data = []
    print "Building TaggedDocuments"
    total = len(data_df[['title', 'plot']].as_matrix().tolist())
    processed = 0
    for x in data_df[['title', 'plot']].as_matrix().tolist():
        label = ["_".join(x[0].split())]
        words = []
        sentences = sent_tokenize(x[1].decode("utf8"))
        for s in sentences:
            words.extend([x.lower() for x in word_tokenize(s)])
        doc = TaggedDocument(words, label)
        data.append(doc)

        processed += 1
        if processed % 10000 == 0:
            print processed, "/", total

    model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=2)
    print "Building Vocabulary"
    model.build_vocab(data)

    for epoch in range(20):
        print "Training epoch %s" % epoch
        model.train(data)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.train(data)

    # Build doc2vec vectors
    x_data = []
    y_data = []
    genres = data_df.drop(['title', 'plot', 'plot_lang'], axis=1).as_matrix()
    names = data_df[['title']].as_matrix().tolist()
    for i in range(len(names)):
        name = names[i][0]
        label = "_".join(name.split())
        x_data.append(model.docvecs[label])
        y_data.append(genres[i])

    return np.array(x_data), np.array(y_data)


#
# train classifiers and argument handling
#

def train_test_svm(x_data, y_data, genres):

    stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33)
    for train_index, test_index in stratified_split.split(x_data, y_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

    """
    print "LinearSVC"
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
    ])
    parameters = {
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    grid_search(x_train, y_train, x_test, y_test, genres, parameters, pipeline)

    print "LogisticRegression"
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
    ])
    parameters = {
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    grid_search(x_train, y_train, x_test, y_test, genres, parameters, pipeline)
    """

    print "LinearSVC"
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(SVC(), n_jobs=1)),
    ])
    """
    parameters = {
        "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__class_weight": ['balanced', None],
    }
    """
    parameters = [

        {'clf__estimator__kernel': ['rbf'],
         'clf__estimator__gamma': [1e-3, 1e-4],
         'clf__estimator__C': [1, 10]
        },

        {'clf__estimator__kernel': ['poly'],
         'clf__estimator__C': [1, 10]
        }
         ]

    grid_search(x_train, y_train, x_test, y_test, genres, parameters, pipeline)


def grid_search(train_x, train_y, test_x, test_y, genres, parameters, pipeline):
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    grid_search_tune.fit(train_x, train_y)

    print
    print("Best parameters set:")
    print grid_search_tune.best_estimator_.steps
    print

    # measuring performance on test set
    print "Applying best classifier on test data:"
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)

    print classification_report(test_y, predictions, target_names=genres)


def parse_arguments():
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        '--clf', dest='classifier', choices=['nb', 'linearSVC', 'logit'])

    arg_parser.add_argument(
        '--vectors', dest='vectors', type=str, choices=['tfidf', 'word2vec', 'doc2vec'])

    return arg_parser, arg_parser.parse_args()


def main():
    args_parser, args = parse_arguments()

    if len(sys.argv) == 1:
        args_parser.print_help()
        sys.exit(1)

    # load pre-processed data
    print "Loading already processed training data"
    data_df = pd.read_csv("movies_genres_en.csv", delimiter='\t')
    # all the list of genres to be used by the classification report
    genres = list(data_df.drop(['title', 'plot', 'plot_lang'], axis=1).columns.values)

    if args.vectors == 'tfidf':

        # split the data, leave 1/3 out for testing
        data_x = data_df[['plot']].as_matrix()
        data_y = data_df.drop(['title', 'plot', 'plot_lang'], axis=1).as_matrix()
        stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.33)
        for train_index, test_index in stratified_split.split(data_x, data_y):
            x_train, x_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]

        # transform matrix of plots into lists to pass to a TfidfVectorizer
        train_x = [x[0].strip() for x in x_train.tolist()]
        test_x = [x[0].strip() for x in x_test.tolist()]

        if args.classifier == 'nb':
            # MultinomialNB: Multi-Class OneVsRestClassifier
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
            parameters = {
                'tfidf__max_df': (0.25, 0.5, 0.75),
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'clf__estimator__alpha': (1e-2, 1e-3)
            }
            grid_search(train_x, y_train, test_x, y_test, genres, parameters, pipeline)
            exit(-1)

        if args.classifier == 'linearSVC':
            # LinearSVC
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
            parameters = {
                'tfidf__max_df': (0.25, 0.5, 0.75),
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                "clf__estimator__C": [0.01, 0.1, 1],
                "clf__estimator__class_weight": ['balanced', None],
            }
            grid_search(train_x, y_train, test_x, y_test, genres, parameters, pipeline)
            exit(-1)

        if args.classifier == 'logit':
            # LogisticRegression
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])
            parameters = {
                'tfidf__max_df': (0.25, 0.5, 0.75),
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                "clf__estimator__C": [0.01, 0.1, 1],
                "clf__estimator__class_weight": ['balanced', None],
            }
            grid_search(train_x, y_train, test_x, y_test, genres, parameters, pipeline)
            exit(-1)

    if args.vectors == 'word2vec':
        if os.path.exists("pos_tagged_data.dat"):
            print "Loading Part-of-Speech tagged data..."
            with open('pos_tagged_data.dat', 'rb') as f:
                data = pickle.load(f)
                x_data, y_data = data[0], data[1]
        else:
            print "Part-of-Speech tagging..."
            x_data, y_data = post_tag_documents(data_df)
            with open('pos_tagged_data.dat', 'w') as f:
                pickle.dump((x_data, y_data), f)

        pos_filter = ['NOUN', 'ADJ']

        # get embeddings for train and test data
        x_embeddings = word2vec(x_data, pos_filter)

        # need to transform back into numpy array to apply StratifiedShuffleSplit
        y_data = np.array(y_data)

        train_test_svm(x_embeddings, y_data, genres)
        exit(-1)

    if args.vectors == 'doc2vec':
        if os.path.exists("doc2vec_data.dat"):
            print "Loading Doc2Vec vectors"
            with open('doc2vec_data.dat', 'rb') as f:
                data = pickle.load(f)
                x_data, y_data = data[0], data[1]
        else:
            print "Generating Doc2Vec vectors"
            x_data, y_data = doc2vec(data_df)
            with open('doc2vec_data.dat', 'w') as f:
                pickle.dump((x_data, y_data), f)

        train_test_svm(x_data, y_data, genres)
        exit(-1)


if __name__ == "__main__":
    main()
