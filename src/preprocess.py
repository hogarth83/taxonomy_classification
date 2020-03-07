import argparse
from core import normalize_corpus, prepare_taxonomy_classes
import spacy
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel

# initialize pandarallel
pandarallel.initialize()

# load spacy model
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)


# prepare stopword dictionary
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
spacy_stopwords.remove('no')
spacy_stopwords.remove('not')


# keep samples which has higher probability than prob_threshold
prob_threshold = 0.5


def get_label_and_score(row, prob_threshold=0.5):
    """
    parsing list with label and score: get label if the score is higher than prob_threshold

    :param row: pandas row
    :param prob_threshold: float
    :return:
        label: list
    """
    label = []

    for item, score in row:
        if score > prob_threshold:
            label.append('_'.join(item.replace(',', ' ').split()))

    if label != []:
        return label
    else:
        return []


def feature_process(row, nlp, spacy_stopwords, prob_threshold):
    """

    :param row: pandas row
        One row from pandas dataframe
    :param nlp: spacy model
        Spacy model
    :param spacy_stopwords: dict
        Stopword dictionary
    :param prob_threshold: float
        Thershold for filter classes which has a low probalbility based on previous model output
    :return:
        item_line: pandas row
            Pandas row with cleared text
    """
    item_line = {}
    label = get_label_and_score(row.labels, prob_threshold=prob_threshold)

    if label != []:
        item_line['doc_label'] = label
        text = ' '.join([row.content['title'], row.content['fullTextHtml']])
        item_line['doc_token'] = normalize_corpus(text, nlp, spacy_stopwords).split()
        item_line['doc_keyword'] = []
        item_line['doc_topic'] = []

    return item_line


def feature_process_test(row, nlp, spacy_stopwords):
    """"""
    item_line = {}

    item_line['doc_label'] = ['dummy']
    text = ' '.join([row.content['title'], row.content['fullTextHtml']])
    item_line['doc_token'] = normalize_corpus(text, nlp, spacy_stopwords).split()
    item_line['doc_keyword'] = []
    item_line['doc_topic'] = []

    return item_line


def feature_generator(odir, trname):
    """Impute missing values."""

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # read train and test json in pandas dataframe
    train = pd.read_json(trname)


    # clean them
    print('Processing on train set')
    train['feature'] = train.parallel_apply(feature_process, args=[nlp, spacy_stopwords, prob_threshold], axis=1)
    print('Done!')


    # remove empty dict
    train = train[train['feature'].str.len() > 1]

    # split to train and dev
    train, test = train_test_split(train, test_size=1 - train_ratio, random_state=42)
    dev, test = train_test_split(test, test_size=test_ratio / (test_ratio + validation_ratio), random_state=42)

    # save into json
    train['feature'].to_json(os.path.join(odir, 'train_feature.json'), orient='records', lines=True)
    test['feature'].to_json(os.path.join(odir, 'test_feature.json'), orient='records', lines=True)
    dev['feature'].to_json(os.path.join(odir, 'dev_feature.json'), orient='records', lines=True)


def prepare_test_file(odir, trname):
    """
    Prepare test file to make a prediction on it
    :param odir: str
        path to folder where json file will saved
    :param trname: str
        name of input json file
    :return:
        write preprocessed json file into odir
    """

    # read json file
    test = pd.read_json(trname)

    # clean the text
    test['feature'] = test.apply(feature_process_test, args=[nlp, spacy_stopwords], axis=1)

    # write to json into odir
    test['feature'].to_json(os.path.join(odir, 'actual_test_feature.json'), orient='records', lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", dest="odir",
                        required=True, help="output directory")
    parser.add_argument("-r", "--train", dest="trname",
                        required=True, help="training file")
    parser.add_argument("-t", "--test_train", dest="test_train",
                        required=True, help="training file")

    args = parser.parse_args()

    if args.test_train == 'train':
        feature_generator(args.odir, args.trname)
        prepare_taxonomy_classes()
    elif args.test_train == 'test':
        prepare_test_file(args.odir, args.trname)
    else:
        raise ValueError("It must be 'train' or 'test', please specify it")

