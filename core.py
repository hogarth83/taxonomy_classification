import wget
import re
from bs4 import BeautifulSoup
import unicodedata
from src.contractions import CONTRACTION_MAP
import json

def data_loader():
    """
    Load data from Google Drive
    :return:
    Share data into /data folder: predict_paylaod.json, taxonomy_mappings.json, train_data.json
    """

    # load data from google drive
    wget.download('https://docs.google.com/uc?export=download&id=1a0dhmA76EvimwbHCWtcj1J1yqnLRu2Ec', 'data/predict_paylaod.json')
    wget.download('https://docs.google.com/uc?export=download&id=1zSbSvUt-qeiv0JXD3zbmZQiw7xOb22E1', 'data/taxonomy_mappings.json')
    wget.download('https://docs.google.com/uc?export=download&id=1LP7k3aBR34L0UjsvDqUSa7owyYmG9EM7', 'data/train_data.json')


def prepare_taxonomy_classes():
    """
    Prepare taxonomy hierarchy to Hierarchical Text Classification Algorithm. Find root and child within categories.
    Read data/taxonomy_mappings.json and modified names, find root and child save features/news.taxonomy
    The modified names saved as a mappaing dictionary to identify the classes during inference time (data/modified_taxonomy.json)

    :return:

    """
    output_json = json.load(open('data/taxonomy_mappings.json'))

    roots = []
    items = []

    modified_taxonomy = {}

    for key, v in output_json.items():
        if '>' not in v:
            roots.append('_'.join(v.replace(',', ' ').split()))
        else:
            items.append('_'.join(v.replace(',', ' ').split()))

        modified_taxonomy['_'.join(v.replace(',', ' ').split())] = int(key)

    with open('data/modified_taxonomy.json', 'w') as fp:
        json.dump(modified_taxonomy, fp)
    fp.close()

    with open('features/news.taxonomy', 'w') as f:
        f.write('{}'.format('\t'.join(['Root'] + roots)))
        for root in roots:
            root_item = [x for x in items if root in x.split('>')[0]]
            f.write('\n{}'.format('\t'.join([root] + root_item)))
    f.close()


def remove_stopwords(text, nlp, spacy_stopwords, is_lower_case=False):
    """
    Remove stopword from text
    :param text: string
    :param spacy_stopwords: dict
    :param nlp: spacy model
    :param is_lower_case: boolean
    :return:
        text: string
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]

    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in spacy_stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in spacy_stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def lemmatize_text(text, nlp):
    """
    Lemmatize text

    :param text: str
    :param nlp: spacy model
    :return:
        text: str
    """

    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_special_characters(text, remove_digits=False):
    """
    Remove digits from text

    :param text: str
    :param remove_digits: boolean
    :return:
        text: str
    """

    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    """
    Expand contraction: e.g. can't -> can not

    :param text: str
    :param contraction_mapping: dict
    :return:
        text: str
    """

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_accented_chars(text):
    """
    Remove accented chars

    :param text: str
    :return:
        text: str
    """

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def strip_html_tags(text):
    """
    Clean html: remove html tags

    :param text: str
    :return:
        text: str
    """

    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def normalize_corpus(doc, nlp, spacy_stopwords, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True, remove_digits=True):
    """
    Normalize corpus

    :param doc: text
        Input text file
    :param html_stripping: boolean
        stripping html file
    :param contraction_expansion: boolean
        expand contraction
    :param accented_char_removal: boolean
        remove accented chars
    :param text_lower_case: boolean
        takes chars lower cases
    :param text_lemmatization: boolean
        lemmatize text
    :param special_char_removal: boolean
        remove special chars
    :param stopword_removal: boolean
        remove stopwords
    :param remove_digits: boolean
        remove digits
    :return:
        doc: str
            normalized text
    """

    # strip HTML
    if html_stripping:
        doc = strip_html_tags(doc)
    # remove accented characters
    if accented_char_removal:
        doc = remove_accented_chars(doc)
    # expand contractions
    if contraction_expansion:
        doc = expand_contractions(doc)
    # lowercase the text
    if text_lower_case:
        doc = doc.lower()
    # remove extra newlines
    doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
    # lemmatize text
    if text_lemmatization:
        doc = lemmatize_text(doc, nlp)
    # remove special characters and\or digits
    if special_char_removal:
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        doc = remove_special_characters(doc, remove_digits=remove_digits)
        # remove extra whitespace
    doc = re.sub(' +', ' ', doc)
    # remove stopwords
    if stopword_removal:
        doc = remove_stopwords(doc, nlp, spacy_stopwords, is_lower_case=text_lower_case)

    return doc