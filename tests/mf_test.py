import csv
import logging
import math
import random
import unittest

import numpy as np
from sklearn.model_selection import KFold

import acmf.core.constants as const
import acmf.core.logconfig as conf
from acmf.rcm.acrmf import ACRMF
from acmf.rcm.armf import AMF
from acmf.rcm.crmf import CMF
from acmf.rcm.rmf import MF

random.seed(123)
np.random.seed(123)


class MfTest(unittest.TestCase):
    """
    Matrix Factorization test case
    """

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        conf.set_log_config(logging.DEBUG, True)
        # path = '../dataset/test_dt_ele_1/Electron_ITEM_50_71_spam_filter_1_FO_1.dt'
        path = '../dataset/test_dt_baby_1/BABY_ITEMS_32_69_spam_filter_1_FO_1_test_out_1.dt'
        csv_reader = csv.reader(open(path), delimiter=',')
        data = [row for row in csv_reader if len(row) >= 2]
        self.data = data[1:]
        self.user_id_index = 1
        self.item_id_index = 2
        self.rating_index = 3
        self.aspect_data_index = 4
        self.context_data_index = 5
        self.context_info = (5, 4)
        self.factors = 15
        self.epochs = 30
        self.alpha = 0.01
        self.beta = 0.001

    def get_mf_conf(self):
        return {
            "user_id": self.user_id_index,
            "item_id": self.item_id_index,
            "rating": self.rating_index,
            const.ALPHA: self.alpha,
            const.BETA: self.beta
        }

    def test_mf(self):
        mf = MF(self.data, self.data, self.get_mf_conf(), self.factors, self.epochs)
        mf.train(2)
        # mf.plot()
        pass

    def get_cmf_conf(self):
        return {
            "user_data_index": self.user_id_index,
            "item_data_index": self.item_id_index,
            "rating_data_index": self.rating_index,
            "context_data_index": self.context_data_index,
            'iter': self.epochs,
            'mf_rank': self.factors,
            'context_info': self.context_info,
            const.ALPHA: self.alpha,
            const.BETA: self.beta
        }

    def test_cmf(self):
        model = CMF(self.data, self.data, self.get_cmf_conf())
        model.train(2)
        model.plot()

    def get_amf_conf(self):
        return {
            const.CONFIG_DATA_INDEX_USER: self.user_id_index,
            const.CONFIG_DATA_INDEX_ITEM: self.item_id_index,
            const.CONFIG_DATA_INDEX_RATING: self.rating_index,
            const.CONFIG_DATA_INDEX_FEATURES: self.aspect_data_index,
            const.MF_RANK_RATING: self.factors,
            const.ITERATIONS: self.epochs,
            const.ALPHA: self.alpha,
            const.BETA: self.beta
        }

    def test_amf(self):
        model = AMF(self.data, self.data, self.get_amf_conf())
        model.train(1)

    def test_check(self):
        mf = MF(self.data, self.data, self.get_mf_conf(), self.factors, self.epochs)
        mf.train(5)
        model = AMF(self.data, self.data, self.get_amf_conf())
        model.train(5)

    def get_acmf_conf(self):
        return {
            const.CONFIG_DATA_INDEX_USER: self.user_id_index,
            const.CONFIG_DATA_INDEX_ITEM: self.item_id_index,
            const.CONFIG_DATA_INDEX_RATING: self.rating_index,
            const.CONFIG_DATA_INDEX_FEATURES: self.aspect_data_index,
            const.CONFIG_DATA_INDEX_CONTEXT: self.context_data_index,
            const.CONFIG_CONTEXT_INFO: self.context_info,
            const.MF_RANK_RATING: self.factors,
            const.ITERATIONS: self.epochs,
            const.ALPHA: self.alpha,
            const.BETA: self.beta
        }

    def test_acmf(self):
        model = ACRMF(self.data, self.data, self.get_acmf_conf())
        model.train(2)

    def test_all_mf(self):
        print_epochs = 2
        logging.info("----------- MF -------------")
        mf = MF(self.data, self.data, self.get_mf_conf(), self.factors, self.epochs)
        mf.train(print_epochs)
        logging.info("----------- CMF -------------")
        cmf = CMF(self.data, self.data, self.get_cmf_conf())
        cmf.train(print_epochs)
        logging.info("----------- AMF -------------")
        amf = AMF(self.data, self.data, self.get_amf_conf())
        amf.train(print_epochs)
        logging.info("----------- ACRMF -------------")
        acrmf = ACRMF(self.data, self.data, self.get_acmf_conf())
        acrmf.train(print_epochs)

    def get_folds(self, splits):
        fold = KFold(n_splits=splits)
        x = []
        y = []
        for row in self.data:
            x.append([row[1], row[2]])
            y.append(row[3])
        return fold.split(x, y)

    def get_folds1(self, splits):
        dicts = {}
        for i, row in enumerate(self.data):
            item = row[2]
            indexes = dicts.get(item)
            if indexes is None:
                indexes = []
            indexes.append(i)
            dicts[item] = indexes

        out_ls = []
        indexes_ls = dicts.values()
        for ls in indexes_ls:
            random.shuffle(ls)
            inner_ls = []
            l = len(ls)
            div = int(math.ceil(l / splits))
            for i in range(0, l, div):
                inner_ls.append(ls[i:i + div])
            out_ls.append(inner_ls)

        folds = []
        for i in range(splits):
            folds.append(self.get_fold_data(i, out_ls))
        return folds

    def get_fold_data(self, index, data):
        train, test = set(), set()
        for data_ls in data:
            if len(data_ls) == 1:
                for ls in data_ls:
                    for l in ls:
                        train.add(l)
            else:
                for i, ls in enumerate(data_ls):
                    if i == index:
                        for l in ls:
                            test.add(l)
                    else:
                        for l in ls:
                            train.add(l)
        return train, test

    def get_folds2(self, per):
        user_set = set()
        item_set = set()
        for i, row in enumerate(self.data):
            user = row[1]
            item = row[2]
            user_set.add(user)
            item_set.add(item)

        user_ls = [u for u in user_set]
        item_ls = [i for i in item_set]

        random.shuffle(user_ls)
        random.shuffle(item_ls)

        user_split = int(len(user_ls) * per)
        item_split = int(len(item_ls) * per)
        train, test = self.get_data_by_ls(
            user_ls[user_split:],
            item_ls[item_split:],
            user_ls[0:user_split],
            item_ls[0:item_split],
        )
        return [(train, test)]

    def get_data_by_ls(self, train_user_ls, train_item_ls, test_user_ls, test_item_ls):
        train_index = []
        test_index = []
        for i, row in enumerate(self.data):
            user = row[1]
            item = row[2]
            if user in train_user_ls and item in train_item_ls:
                train_index.append(i)
            elif user in test_user_ls and item in test_item_ls:
                test_index.append(i)
        return train_index, test_index

    @staticmethod
    def avg_by_index(list, index):
        sum = 0
        count = 0
        for l in list:
            data = l[index]
            sum += data
            if data != 0:
                count += 1
        if sum == 0:
            return 0
        return sum / count

    @staticmethod
    def print_list(list):
        out = ""
        for f in list:
            out += ('%0.3f' % f) + ", "
        return out.strip()

    def test_kfold1(self):
        train, test = self.get_folds2()
        print(len(train), len(test))

    def inner_test(self):
        print_epochs = 5
        all_test = []
        for t in range(2):
            logging.info('ROUND:' + str(t))
            stats = []
            i = 0
            for train, test in self.get_folds1(3):
                fold_sts = []
                logging.info("---- FOLD ----" + str(i))
                logging.info("{},{}".format(len(train), len(test)))
                train_data = [self.data[index] for index in train]
                test_data = [self.data[index] for index in test]

                logging.info("-- MF -->")
                mf = MF(train_data, test_data, self.get_mf_conf(), self.factors, self.epochs)
                mf.train(print_epochs)
                fold_sts.append(mf.error(train_data))
                fold_sts.append(mf.error(test_data))

                logging.info("-- CMF -->")
                cmf = CMF(train_data, test_data, self.get_cmf_conf())
                cmf.train(print_epochs)
                fold_sts.append(cmf.error(train_data))
                fold_sts.append(cmf.error(test_data))

                logging.info("-- AMF -->")
                amf = AMF(train_data, test_data, self.get_amf_conf())
                amf.train(print_epochs)
                fold_sts.append(amf.error(train_data))
                fold_sts.append(amf.error(test_data))

                logging.info("-- ACMF -->")
                acmf = ACRMF(train_data, test_data, self.get_acmf_conf())
                acmf.train(print_epochs)
                fold_sts.append(acmf.error(train_data))
                fold_sts.append(acmf.error(test_data))

                stats.append(fold_sts)
                i += 1
            all_test.append(stats)
        return all_test

    def avg_tuple(self, f):
        # tuple to list
        ls = []
        for f1 in f:
            ls1 = []
            for f11 in f1:
                ls1.append(f11[0])
                ls1.append(f11[1])
            ls.append(ls1)
        return [MfTest.avg_by_index(ls, i) for i in range(len(ls[0]))]

    def test_kfold(self):
        all_test = self.inner_test()
        avg_ls = []
        for f in all_test:
            logging.info("------------")
            for fi in f:
                logging.info(fi)
            avg_ls.append(self.avg_tuple(f))

        logging.info("---------------")
        for avg in avg_ls:
            logging.info(avg)

        logging.info("--------------")
        logging.info([MfTest.avg_by_index(avg_ls, i) for i in range(len(avg_ls[0]))])
        #     avg_list = [MfTest.avg_by_index(stats, i) for i in range(len(stats[0]))]
        #     test_dict = {
        #         'folds': stats,
        #         'avg': avg_list
        #     }
        #     all_test.append(test_dict)
        #
        # avg_all = [t['avg'] for t in all_test]
        # all_avg_list = [MfTest.avg_by_index(avg_all, i) for i in range(len(avg_all[0]))]
        #
        # for test_dict in all_test:
        #     print("------------->")
        #     sts = test_dict['folds']
        #     for s in sts:
        #         print(MfTest.print_list(s))
        #     print("-------------")
        #     print(MfTest.print_list(test_dict['avg']))
        #
        # print("------------------------------------\n\n")
        # for test_dict in all_test:
        #     print(MfTest.print_list(test_dict['avg']))
        #
        # print(MfTest.print_list(all_avg_list))


pass
