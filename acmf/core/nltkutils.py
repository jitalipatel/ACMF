import logging
import re

import nltk
from nltk.corpus import wordnet

FILTERS_CONST = ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'NN', 'NNS')


def get_sentences(text):
    return [text_line for text_line in nltk.sent_tokenize(text)]


def get_pos_from_text(text):
    return [nltk.pos_tag(nltk.word_tokenize(text))]


def filter(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'( )+', ' ', text)
    text = text.strip()
    return text


def get_synsets(text):
    out_text = ''
    for t in text.split(" "):
        out_word = t
        synsets = wordnet.synsets(t)
        if len(synsets) > 0:
            out_word = synsets[0].lemmas()[0].name()
        out_text += out_word + ' '
    return out_text.strip()


def get_pos_tags(text):
    """
    1. Convert into sentences
    2. Word tokenizing of each sentence
    3. POS tagging
    :param text: sentences
    :return: list
    """
    try:
        ls = [nltk.pos_tag(nltk.word_tokenize(text_line)) for text_line in nltk.sent_tokenize(text)]
    except Exception as e:
        print("Text:" + text)
        print(e)
        raise
    # logging.debug('POS Tagging:' + str(ls))
    return ls


def get_pos_tags_filters(text):
    ls = get_pos_tags(str(text))
    out_text = ""
    for l in ls:
        for pos in l:
            if pos[1] in FILTERS_CONST:
                out_text += " " + pos[0]
    return out_text


def get_pos_tags_filters_ls(text_ls):
    length = len(text_ls)
    intr = length / 100
    intr = int(intr)
    ls = []
    i = 0
    for text in text_ls:
        ls.append(get_pos_tags_filters(text))
        i += 1
        if i % intr == 0:
            logging.info(i)
    return ls
