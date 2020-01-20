from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import pickle
import nltk
import sys
from config import *


def load_data(database_filepath):
    """
    load data from the database

    :param database_filepath: path to the database
    :return:
    X: features (pandas.DataFrame)
    Y: targets (pandas.DataFrame)
    category_names: targets names (list)
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con=engine)

    # feature is the string msg, target is the one hot encoded categories
    X, Y = df['message'], df.iloc[:, 4:]

    # "related" column has 0, 1, 2 values
    # change 2 to 1, so that the value is only 1 or 0
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    tokenize the text into tokens

    :param text: raw text (string)
    :return: tokens (list of strings)
    """
    # remove the special chars
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)  # function in the nltk lib

    # remove the stopwords
    words = [word for word in words if word not in stopwords.words("english")]

    # lemmatize the verbs
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]
    return words


def build_model():
    """
    build the processing pipelines

    :return: a pipeline with grid search
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0)
    }

    model = GridSearchCV(estimator=pipeline,
                         param_grid=parameters,
                         verbose=3,
                         cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the performance of the model

    :param model: machine learning pipeline
    :param X_test: test dataset for features
    :param Y_test: test dataset for targets
    :param category_names: targets names
    :return: None
    """
    y_pred = model.predict(X_test)
    logger.info("\n" + classification_report(Y_test.values, y_pred, target_names=category_names))
    logger.info('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    """
    save the model to a file

    :param model: machine learning pipeline
    :param model_filepath: path to file
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def train_save_classifier():
    """
    main processing function
    """
    logger.info('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    logger.info('Building model...')
    model = build_model()

    logger.info('Training model...')
    model.fit(X_train, Y_train)

    logger.info('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    logger.info('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    logger.info('Trained model saved!')


if __name__ == "__main__":
    if len(sys.argv) == 3:
        nltk.download(['punkt', "stopwords", "wordnet"])
        database_filepath, model_filepath = sys.argv[1:]
        train_save_classifier()
    else:
        logger.error(
            'Please provide the filepath of the disaster messages database '
            'as the first argument and the filepath of the pickle file to '
            'save the model to as the second argument. \n\nExample: python '
            'train_classifier.py ../data/DisasterResponse.db classifier.pkl'
        )
