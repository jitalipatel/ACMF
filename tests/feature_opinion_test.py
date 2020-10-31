import logging
import re
import unittest
from collections import Counter

import numpy as np
import pandas as pd
from nltk.corpus import stopwords

import acmf.core.coreutils as core_utils
import acmf.core.feature_extraction as fe
import acmf.core.logconfig as conf
import acmf.core.nltkutils as nltk_utl
from acmf.core.feature_extraction import FeatureOpinionExtractor


def scoring(x):
    return 1 / (1 + np.exp(x))


class FeaturePairTest(unittest.TestCase):
    stopwords_set = set([w.lower() for w in stopwords.words('english')])

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        conf.set_log_config(logging.INFO, True)

    def testExtract(self):
        texts = [
            'this product is a too good',
            'camera quality is good',
            'service is extremely bad. display is too large. it create too much noise. compressor is quite heavy. price is also too high'
        ]
        extractor = FeatureOpinionExtractor(texts, window_size=3)
        ls = extractor.process(itrs=1)
        print("Output")
        for l in ls:
            print(l)

    def data_filter(self, data):
        data = data.lower().strip()
        data = re.sub(r"[^\w\s']", '', data)
        words = [w.strip() for w in data.split(" ")]
        data = " ".join([w for w in words if w != '' and w not in FeaturePairTest.stopwords_set])
        data = re.sub(r' \d+ ', ' ', data)
        data = re.sub(r'\n+', ' ', data)
        data = re.sub(r'( )+', ' ', data)
        return data

    def create_cloud(self, data):
        data = self.data_filter(data)
        count = Counter(data.split(" "))
        count = Counter(dict(filter(lambda x: x[1] >= 15, count.items())))
        return count

    def test2(self):
        path = '../dataset/unit_test/my_set_filters.csv'
        out_path = '../dataset/unit_test/'
        # out_path = '../dataset/Electronics/out/'
        out_name = 'my_set_filters_last_1_cloud'
        df = pd.read_csv(path, encoding='utf-8')
        logging.info("CSV loaded:{}".format(len(df)))

        unique_names = list(set(df['asin'].tolist()))

        df['feature_pair'] = ""

        for unique_name in unique_names:
            df1 = df[unique_name == df['asin']]
            reviews = [str(r) for r in df1['reviewText'].tolist()]
            data = "\n".join(reviews)
            counter = self.create_cloud(data)

            for index, row in df.iterrows():
                if index % 100 == 0:
                    print(index)
                if row['asin'] == unique_name:
                    features = {}
                    new_data = str(row['reviewText'])
                    new_data = self.data_filter(new_data)
                    for w in new_data.split(" "):
                        count = counter[w]
                        if count != 0:
                            old_count = features.get(w)
                            if old_count is None:
                                old_count = count
                            else:
                                old_count += count
                            features[w] = old_count
                    if len(features) > 0:
                        ls = [(key, '', scoring(value)) for key, value in features.items()]
                        df.at[index, 'feature_pair'] = ls

        df['context'] = core_utils.random_context(len(df), [2, 3])
        df[['reviewerID', 'asin', 'overall', 'feature_pair', 'context']].to_csv(out_path + out_name + '.dt')
        df[['reviewText']].to_csv(out_path + out_name + '_all.dt')
        df[['feature_pair']].to_csv(out_path + out_name + '_features.dt')

        pass

    def testDf(self):
        path = '../dataset/test_dt_baby_1/BABY_ITEMS_32_69_spam_filter_1_test_out_1.csv'
        out_path = '../dataset/test_dt_baby_1/'
        out_name = 'BABY_ITEMS_32_69_spam_filter_1_FO_1_test_out_1'
        df = pd.read_csv(path, encoding='utf-8')
        logging.info("CSV loaded:{}".format(len(df)))
        # df = df[:1000]

        # df['unixReviewTime'] = pd.to_datetime(df['unixReviewTime'], unit='s')
        df['unixReviewTime'] = pd.to_datetime(df['unixReviewTime'])
        df['features'] = ""

        df['context'] = core_utils.random_context(len(df), [5, 4])

        # df['feature_pair'] = fe.get_fe_pairs(df['reviewText'])

        extractor = FeatureOpinionExtractor(df['reviewText'], window_size=3)
        df['feature_pair'] = extractor.process(itrs=1000)

        df[['reviewerID', 'asin', 'overall', 'feature_pair', 'context']].to_csv(out_path + out_name + '.dt')
        df[['reviewText']].to_csv(out_path + out_name + '_all.dt')
        df[['feature_pair']].to_csv(out_path + out_name + '_features.dt')

    pass

    def test_remove(self):
        in_path = '../dataset/test_dt_ele_1/Electron_ITEM_EQ_50_spam_filter_1_FO_1.dt'
        out_path = '../dataset/test_dt_ele_1/Electron_ITEM_EQ_50_spam_filter_1_FO_1_remove_700.dt'
        df = pd.read_csv(in_path, encoding='utf-8')

        unique_reviewer_id = list(set(df['reviewerID'].tolist()))

        for unique_name in unique_reviewer_id:
            df1 = df[unique_reviewer_id == df['reviewerID']]
            drop_indices = np.random.choice(df.index, 2000, replace=False)

        # drop_indices = np.random.choice(df.index, 2000, replace=False)
        # df_subset = df.drop(drop_indices)
        # df_subset.to_csv(out_path, index=False)
        pass




    def testDf1(self):
        path = '../dataset/TA_review_gene_DATASET_arr.json'
        out_path = '../dataset/unit_test/'
        df = pd.read_json(path)
        df = df[['User Id', 'Item Id', 'Rating', 'Review']]
        df = df.rename(columns={
            'User Id': 'reviewerID',
            'Item Id': 'asin',
            'Rating': 'overall',
            'Review': 'reviewText'
        })

        df['context'] = core_utils.random_context(len(df), [3, 4])

        extractor = FeatureOpinionExtractor(df['reviewText'], window_size=3)
        df['feature_pair'] = extractor.process(itrs=10)

        df[['reviewerID', 'asin', 'overall', 'feature_pair', 'context']].to_csv(out_path + 'ta_my_set_all_1.dt')
        df[['reviewText']].to_csv(out_path + 'ta_my_set_all_Text_1.dt')
        df[['feature_pair']].to_csv(out_path + 'ta_my_set_all_feature_1.dt')
        pass

    def test_post_filter(self):
        path = '../dataset/unit_test/my_set_all_2.dt'
        out_path = '../dataset/unit_test/'
        df = pd.read_csv(path)

        features = df['feature_pair'].tolist()

        new_features = core_utils.filter_features(features, "../dataset/unit_test/filter_2.txt")

        print("EXIT")

    def testLoadDf(self):
        df = pd.read_csv('../dataset/unit_test/2_Musical_Instruments_all_feature.dt')
        new_df = df[pd.notna(df.feature_pair)]
        ls = new_df.feature_pair.tolist()
        total = 0
        for l in ls:
            total += len(eval(l))

        print(len(df), len(new_df), total)

    pass



    def test_rand_context(self):
        ls = core_utils.random_context1(50, [2, 2])
        pass

    def test_opinion_score(self):
        score = fe.nltk_get_polarity("very")
        print(score)
        print(fe.nltk_get_polarity("good"))
        print(fe.nltk_get_polarity("very good"))
        print(fe.nltk_get_polarity("dkfdnsdnfksdfnsdk"))
        print(fe.nltk_get_polarity(",,,,44343 good"))
        pass

    def test_filter(self):
        text = nltk_utl.filter(
            "Like another reviewer, I was going crazy comparing much more expensive covers - then found this.Figured that at this price point, if I get a year or two out of it, I'd be happy.It seems that it will last a whole lot longer though.It fits very well, and the bottom has a LOT of ventilation holes to keep things cool.I really like the prop-up &#34;feet&#34; on the bottom - makes it much easier for me to type at a desk, etc.The keyboard cover wasn't so great - fairly thick and rubbery and interfered with typing. I bought different different keyboard cover that's much better.")
        print(text)
        pass

    def test_word_net(self):
        print(nltk_utl.get_synsets("THIS IS TEXT"))
        pass
