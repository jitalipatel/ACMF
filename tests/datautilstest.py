import logging
import unittest

import pandas as pd

import acmf.core.datautils as du
import acmf.core.logconfig as conf


class DataSetUtilsTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        conf.set_log_config(logging.DEBUG, True)

    def test_Merge_Products(self):
        du.merge_products('../dataset/Electronics/parts', '../dataset/Electronics/out/merge_GE_100.csv')

    def test_Merge_Products1(self):
        du.merge_products1('../dataset/baby/part_items/',
                           '../dataset/test_dt_baby_1/BABY_ITEMS_32_69.csv', lambda x: x == 32)

    # ---- ELE ----
    # 43 - .48
    # 46 - .58
    # 50 - .71

    # ---- BABY ----
    # 25 - 49
    # 29 - 58
    # 32 - 69


    def test_convert_to_json_array(self):
        du.convert_to_json_array(
            '../dataset/baby/reviews_Baby.json',
            '../dataset/baby/reviews_Baby_arr.json'
        )

    def test_extract_products(self):
        du.extract_products(
            '../dataset/baby/reviews_Baby_arr.json',
            '../dataset/baby/part_items/'
        )
        pass

    def test_write_csv(self):
        du.write_csv(
            '../dataset/Electronics/array/Electronics_5_arr.json',
            '../dataset/Electronics/array/Electronics_5_arr.csv'
        )
        pass

    def test_random_generate(self):
        du.random_generate(
            '../dataset/Electronics/out/merge_GE_USER_GE_50_all_1.dt',
            '../dataset/Electronics/out/merge_GE_USER_GE_50_random_1.dt'
        )

    def test_generate_movies_len(self):
        du.generate_movies_len(
            '/home/divyesh/Documents/ml-1m/ratings.dat',
            '../dataset/Electronics/out/merge_GE_USER_GE_50_all_1.dt',
            '../dataset/Electronics/out/merge_GE_USER_GE_50_len_1.dt'
        )

    def test_generate_travel_dataset(self):
        df = pd.read_csv('../dataset/travel_data.csv')
        df1 = df[['UserID', 'ItemID', 'Rating', 'ItemTimeZone', 'TripType']]

        ls1 = list(set(df1['ItemTimeZone'].tolist()))
        ls2 = list(set(df1['TripType'].tolist()))
        print(len(ls1), len(ls2))

        index1 = [ls1.index(d) for d in df1['ItemTimeZone'].tolist()]
        index2 = [ls2.index(d) for d in df1['TripType'].tolist()]

        final_context = list(zip(index1, index2))

        # df[['reviewerID', 'asin', 'overall', 'feature_pair', 'context']].to_csv(out_path + out_name + '.dt')
        df_final = df[['UserID', 'ItemID', 'Rating']]
        df_final = df_final.rename(columns={'UserID': 'reviewerID', 'ItemID': 'asin', 'Rating': 'overall'})

        path = '../dataset/unit_test/my_set_filters_music_test_2_2_10.dt'
        out_path = '../dataset/unit_test/my_set_filters_music_test_2_2_10_1.dt'
        df_op = pd.read_csv(path)
        ls = df_op['feature_pair'].tolist()

        ls.extend(ls)
        ls.extend(ls)
        ls.extend(ls)

        ls = ls[0:len(df_final)]
        df_final['feature_pair'] = ls
        df_final['context'] = final_context
        df_final.to_csv(out_path)
        pass
