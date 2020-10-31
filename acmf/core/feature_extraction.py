import logging
from math import exp

import numpy as np
# if an adjective follows an adverb, then it is an opinion word
from nltk.sentiment import SentimentIntensityAnalyzer

import acmf.core.nltkutils as nu

ADJECTIVE_CONST = ('JJ', 'JJR', 'JJS')
ADVERB_CONST = ('RB', 'RBR', 'RBS')
NOUN_CONST = ('NN', 'NNS')


class FeatureOpinionExtractor:
    """
        This class is used for extraction { FEATURE - OPINION } Pair from Text.
    """

    def __init__(self, data, window_size=4):
        self.window_size = window_size
        self.data = data
        self.total_sentences = 0
        self.total_feature_opinions = 0
        self.total_adverb_adjective = 0
        self.total_adjective = 0
        self.nan_count = 0
        self.total_opinion = 0
        self.zero_score_opinion = 0
        pass

    def process(self, itrs=10):
        logging.info("Total texts:{}".format(len(self.data)))
        out_list = []
        index = 0
        for text in self.data:
            pair_list = []
            if text is np.nan:
                self.nan_count += 1
            else:
                logging.debug("---------------------------------------------------------------------")
                logging.debug("Index:{}".format(index))
                logging.debug("text: " + str(text))
                for sentence in nu.get_sentences(text):
                    logging.debug("----------------------")
                    logging.debug('sentences:' + sentence)
                    for tags in nu.get_pos_from_text(sentence):
                        self.total_sentences += 1
                        ops = self.feature_pair(tags)
                        pair_list.extend(ops)
            if len(pair_list) == 0:
                out_list.append(None)
            else:
                out_list.append(pair_list)
            if index % itrs == 0:
                self.print_statistics(index)
            index += 1
            # out_list.append(None)
        self.print_statistics(index)
        return out_list

    def feature_pair(self, tag_words):
        logging.debug("Tags:{}".format(tag_words))
        # ------------ Opinion Extraction ------------
        # this variable hold all opinions
        opinion_list_all = []

        # Collect Opinions
        index = 0
        opinion_list = []
        adjective_found, adverb_found = False, False
        tag_words_len = len(tag_words)
        while index < tag_words_len:
            tag_word = tag_words[index]
            feature_list_len = len(opinion_list)

            # ADVERB
            if tag_word[1] in ADVERB_CONST:
                # one or more
                if feature_list_len == 0 or tag_words[opinion_list[feature_list_len - 1]][1] in ADVERB_CONST:
                    opinion_list.append(index)
                    adverb_found = True
            elif tag_word[1] in ADJECTIVE_CONST:
                if feature_list_len == 0 or tag_words[opinion_list[feature_list_len - 1]][1] in ADVERB_CONST or \
                        ADJECTIVE_CONST:
                    opinion_list.append(index)
                    adjective_found = True
            else:
                if feature_list_len > 0:
                    # Accept Opinion Phrase
                    if adjective_found and feature_list_len > 1:  #
                        logging.debug("Pattern match:{}".format([tag_words[i] for i in opinion_list]))
                        opinion_list_all.append(opinion_list)
                        if adverb_found:
                            self.total_adverb_adjective += 1
                            logging.debug('ADVERB_ADJECTIVE')
                        else:
                            self.total_adjective += 1
                            logging.debug('ADJECTIVE++')
                    else:
                        logging.debug("Pattern not match:{}".format([tag_words[i] for i in opinion_list]))
                    # Reset for next iteration
                    opinion_list = []
                    adjective_found = adverb_found = False, False
            index += 1

        # Collect last opinion phrase
        if len(opinion_list) > 1 and adjective_found:
            # Accept Opinion Phrase
            opinion_list_all.append(opinion_list)
            logging.debug("Pattern match:{}".format([tag_words[i] for i in opinion_list]))
            if adverb_found and len(opinion_list) > 1:
                self.total_adverb_adjective += 1
                logging.debug('ADVERB_ADJECTIVE')
            else:
                self.total_adjective += 1
                logging.debug('ADJECTIVE++')
        # else:
        #     logging.debug("Pattern not match:{}".format([tag_words[i] for i in opinion_list]))
        # logging.info("Opinions len:{}".format(opinion_list))

        # [(PREV_OP, CURRENT_OP, NEXT_OP), .....]
        total_opinions = len(opinion_list_all)
        opinion_block = []
        prev_opinion = next_opinion = None
        for i in range(total_opinions):
            prev_index = i - 1
            next_index = i + 1
            if prev_index > 0:
                prev_opinion = opinion_list_all[prev_index]
            if next_index < total_opinions:
                next_opinion = opinion_list_all[next_index]
            opinion_block.append((prev_opinion, opinion_list_all[i], next_opinion))

        # Find Features
        feature_opinion_pair = []
        for prev_opinion, current_opinion, next_opinion in opinion_block:
            # find previous
            start_index = 0
            end_index = current_opinion[0] - 1
            if prev_opinion is not None:
                start_index = prev_opinion[len(prev_opinion) - 1] + 1
            # Collect Previous
            noun_list = self.find_feature_prev(tag_words, end_index, start_index)
            if noun_list is not None:
                feature_opinion_pair.append((noun_list, current_opinion))
                continue

            # find Next
            start_index = current_opinion[len(current_opinion) - 1] + 1
            end_index = len(tag_words) - 1
            if next_opinion is not None:
                end_index = next_opinion[len(next_opinion) - 1] - 1
            # collect next
            noun_list = self.find_feature_next(tag_words, start_index, end_index)
            if noun_list is not None:
                feature_opinion_pair.append((noun_list, current_opinion))

        # Prepare output
        out_feature_opinion_pair = []
        for feature, opinion in feature_opinion_pair:
            # Keep in Sorting Order
            feature.sort()
            opinion.sort()

            # Prepare feature word todo: refactoring
            feature_word = ''
            for i in feature:
                feature_word += tag_words[i][0] + ' '
            feature_word = feature_word.strip()

            # Prepare opinion word todo: refactoring
            opinion_word = ''
            for i in opinion:
                opinion_word += tag_words[i][0] + ' '
            opinion_word = opinion_word.strip()

            opinion_word = nu.filter(opinion_word.lower())
            polarity_score = nltk_get_polarity(opinion_word)

            if polarity_score['compound'] == 0.0:
                self.zero_score_opinion += 1
            self.total_opinion += 1

            out_feature_opinion_pair.append(
                (
                    nu.get_synsets(nu.filter(feature_word.lower())),
                    # nu.filter(feature_word.lower()),
                    opinion_word,
                    get_compound_score(polarity_score)
                )
            )
        logging.debug("Feature_Pairs :{}".format(out_feature_opinion_pair))
        self.total_feature_opinions += len(out_feature_opinion_pair)
        return out_feature_opinion_pair

    def find_feature_prev(self, tag_words, end_index, start_index):
        noun_list = []
        count = 0
        noun_found = False
        while end_index >= start_index and (count < self.window_size or noun_found):
            len_noun = len(noun_list)
            if tag_words[end_index][1] in NOUN_CONST:
                if len_noun == 0 or tag_words[noun_list[len_noun - 1]][1] in NOUN_CONST:
                    noun_list.append(end_index)
                    noun_found = True
            else:
                if len_noun > 0:
                    return noun_list
            end_index -= 1
            count += 1
        if len(noun_list) > 0:
            return noun_list
        return None

    def find_feature_next(self, tag_words, start_index, end_index):
        noun_list = []
        count = 0
        noun_found = False
        while start_index <= end_index and (count < self.window_size or noun_found):
            len_noun = len(noun_list)
            if tag_words[start_index][1] in NOUN_CONST and (
                    len_noun == 0 or tag_words[noun_list[len_noun - 1]][1] in NOUN_CONST):
                noun_list.append(start_index)
                noun_found = True
            else:
                if len_noun > 0:
                    return noun_list
            start_index += 1
            count += 1
        if len(noun_list) > 0:
            return noun_list
        pass

    def print_statistics(self, index):
        logging.info(
            "STATISTICS: Index:{}, Sentences:{}, NAN:{}  Pairs:{}, Adverb_Adjective:{}, Adjectives:{}, Opinion:{}, {}"
            .format(index, self.total_sentences, self.nan_count, self.total_feature_opinions,
                    self.total_adverb_adjective,
                    self.total_adjective, self.total_opinion, self.zero_score_opinion))
        pass


def nltk_get_polarity(text):
    """
    Calculate polarity score
    :param text:
    :return: score {dict}
    """
    return SentimentIntensityAnalyzer().polarity_scores(text)


def get_pos_neg(polarity_score):
    # print(polarity_score)
    return polarity_score['pos'], polarity_score['neg']


def get_compound_score(polarity_score):
    return polarity_score['compound']


def tanh(x):
    """
    Hyperbolic Tangent function- Tanh : It’s mathamatical formula is f(x) = 1 — exp(-2x) / 1 + exp(-2x)
    """

    return float((1 - exp(-2 * x))) / (1 + exp(-2 * x))


def score_cal(posNeg):
    return posNeg[0] - posNeg[1]
    # sumPos = posNeg[0] + posNeg[1]
    # if sumPos == 0:
    #     return 0
    # return posNeg[0] / float(sumPos)

# Move to test
# if __name__ == '__main__':
#    ABC()
#    print(get_pos_tags(
#        'I have purchased a number of AmazonBasics HDMI cables and other cables as well, and have yet to encounter any quality issues. I was impressed with the quality of the cable although the braiding on it is quite stiff and less flexible than other cables. While this is a slight bit of a pain, I know the cable will hold up much better and probably last forever without pulling apart at the connection points.Many people are told by sales people and through advertising by high dollar cable manufacturers that you must spend lots of money to get quality cables. This has been misproven time and time again; you do not have to pay $50-100 for a quality HDMI cable. HDMI, ethernet, USB and other cables have one of the highest mark up margins, meaning they cost very little to make and are marked up substantially. Usually, the saying "you get what you pay for" is true - and especially true when it comes to iPhone, iPad and iPod charging cables for some reason. However, lower priced but quality HDMI cables do exist and this one is prime example.The Audio Return feature and quality of both the audio and video signals on this longer cable worked flawlessly for me on both my LED and Plasma televisions and home theater equipment. If you do have any issues, contact Amazon who will send a replacement.'))
