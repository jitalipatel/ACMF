import logging
import unittest

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import acmf.core.logconfig as conf
import acmf.core.nltkutils as nltkutil
from acmf.core.burstpattern import BurstPattern


class BurstPatternTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        conf.set_log_config(logging.DEBUG, True)

    def testPattern(self):
        df = pd.read_csv('../dataset/test_dt_baby_1/BABY_ITEMS_25_49.csv', encoding='utf-8')
        print(len(df))
        out_path = '../dataset/test_dt_baby_1/BABY_ITEMS_25_49_spam_filter_1_test_out_1.csv'
        df['unixReviewTime'] = pd.to_datetime(df['unixReviewTime'], unit='s')  # convert_dates='unixReviewTime'
        s_dates_count = df['unixReviewTime'].value_counts()
        df_dates_count = s_dates_count.to_frame(name='count').reset_index()
        df_dates_count = df_dates_count.rename(columns={'index': 'date'})

        pattern = BurstPattern(df_dates_count, splitter=('1Y', '2M', '4W'))
        matches_dict = pattern.match()
        logging.info("------------ Result ---------------")

        # todo: Consine_Similarity Logic : Move to Source Code
        my_tmp_dict = dict()
        for key, value in matches_dict.items():
            logging.info("Date:{}, Index:{}, PatternInfo:{}".format(df_dates_count.iloc[key]['date'], key, value))
            divider = value[0][0]
            ls = my_tmp_dict.get(divider)
            if ls is None:
                ls = []
            ls.append(df_dates_count.iloc[key]['date'])
            my_tmp_dict[divider] = ls

        group_dates = []
        for divider, date_list in my_tmp_dict.items():
            groups = pd.DataFrame({'dates': date_list}).groupby(pd.Grouper(key='dates', freq=divider))
            group_dates.extend([grp_data for grp_name, grp_data in groups])

        indexes = []
        all_indexes = []
        for group_date in group_dates:
            inner_indexes = []
            for dates in group_date['dates']:
                inner_index = df[df['unixReviewTime'] == dates].index
                inner_indexes.extend(inner_index.tolist())
                all_indexes.extend(inner_index.tolist())
            indexes.append(inner_indexes)

        logging.info("Indexes...........")
        for index in indexes:
            logging.info(len(index))

        df['filter_text'] = ""
        df['matches'] = True
        logging.info("All indexes............")
        logging.info(len(all_indexes))
        logging.info("POS Tagging....")
        data_ls = nltkutil.get_pos_tags_filters_ls(df.iloc[all_indexes].reviewText)
        df.filter_text.iloc[all_indexes] = data_ls

        logging.info("Co_sine distance Matching....")

        # todo: Slow....
        match_locs = set()
        for index in indexes:
            text = df.iloc[index].filter_text.tolist()
            logging.info(len(text))
            if len(text) > 0:
                vec_array = CountVectorizer().fit_transform(text).toarray()
                for i, i_vec in enumerate(vec_array):
                    logging.info("Compare of i={}".format(i))
                    for j, j_vec in enumerate(vec_array):
                        if i != j:
                            simi = cosine_similarity([i_vec], [j_vec])[0][0]
                            # logging.info(str(i) + " " + str(j) + " " + str(simi))
                            if simi >= 0.50:
                                match_locs.add(index[i])
                                match_locs.add(index[i])
            pass

        df.matches.iloc[list(match_locs)] = False
        new_Df = df[df.matches]
        logging.info(str(len(df)) + "," + str(len(new_Df)))
        new_Df.to_csv(out_path, index=False)
        logging.info("EXIT")

    def test_info(self):
        df = pd.read_csv('../dataset/Electronics/out/Electron_GE_USER_GE_100_115.csv', encoding='utf-8')
        df['unixReviewTime'] = pd.to_datetime(df['unixReviewTime'], unit='s')  # convert_dates='unixReviewTime'
        df = df.set_index('unixReviewTime')
        print(df.index.min())
        print(df.index.max())
        pass
