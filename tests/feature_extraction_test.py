import acmf.core.feature_extraction as feature
import acmf.dataset.dataset_config as config
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import acmf.core.nltkutils as nltk_utils

def write_pair(list_reviews):
    count = 0
    out_text = ''
    for i, text in enumerate(list_reviews):
        opinions_text = ''
        for tags in nltk_utils.get_pos_tags(text):
            opinions = feature.feature_pair(tags)
            if len(opinions) > 0:
                count += 1
                opinions_text += str(opinions) + '\t'
        if len(opinions_text) > 0:
            out_text += str(i) + "-----------> " + '\n' + text + '\n'
            out_text += str(opinions_text) + '\n'
    open('abc.txt', 'w').write(out_text)
    print("Count", len(list_reviews), count)


def write_pair_all(list_reviews):
    out_text = ''
    for i, text in enumerate(list_reviews):
        out_text += " ---------->" + text + '\n'
        for tags in nltk_utils.get_pos_tags(text):
            out_text += str(tags) + '\n'
            opinions = feature.feature_pair(tags)
            out_text += str(opinions) + '\n'
    open('abcd.txt', 'w').write(out_text)


def single_debug():
    text = "Speed seems great as well, no complaints."
    for tags in nltk_utils.get_pos_tags(text):
        feature.feature_pair(tags)


def nltk_polarity(text):
    sia = SentimentIntensityAnalyzer()
    print(sia.polarity_scores(text))
    pass





if __name__ == '__main__':
    # single_debug()
    nltk_polarity('super fast')
    df = pd.read_json(config.get_file_path())
    list_reviews = df.reviewText.tolist()
    write_pair(list_reviews)
    # index_list = [69, 4, 9, 31, 56, 66, 76, 105]
    # sub_list = [text for i, text in enumerate(list_reviews) if i in index_list]
    # write_pair_all(sub_list)
    pass
