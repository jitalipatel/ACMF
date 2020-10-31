import csv
import logging

import numpy as np

import acmf.core.constants as const
import acmf.core.coreutils as core_utils
from acmf.rcm.base_mf import BaseMF

np.random.seed(123)


class AMF(BaseMF):

    def __init__(self, data, test, config):
        self.data = data
        self.test = test
        self.itrs = config[const.ITERATIONS]
        self.alpha = config[const.ALPHA]
        self.beta = config[const.BETA]
        super().__init__(self.itrs, data, test)
        logging.info("Data Size:{}".format(len(self.data)))

        self.data_index_user = config[const.CONFIG_DATA_INDEX_USER]
        self.data_index_item = config[const.CONFIG_DATA_INDEX_ITEM]
        self.data_index_rating = config[const.CONFIG_DATA_INDEX_RATING]
        self.data_index_features = config[const.CONFIG_DATA_INDEX_FEATURES]
        self.mf_rank_rating = config[const.MF_RANK_RATING]

        # Count USER, ITEMS and rating list
        user_set, item_set, feature_set, rating_ls = set(), set(), set(), []
        for row in self.data:
            if len(row) > 2:
                user_set.add(row[self.data_index_user])
                item_set.add(row[self.data_index_item])
                feature_pairs = row[self.data_index_features]
                if feature_pairs != '':
                    list_features = eval(row[self.data_index_features])
                    for feature_data in list_features:
                        feature_set.add(feature_data[0])
                rating = int(row[self.data_index_rating])
                if rating != 0:
                    rating_ls.append(rating)
                pass
            else:
                logging.info("Invalid Row:{}".format(row))

        user_count, item_count, feature_count = len(user_set), len(item_set), len(feature_set)

        logging.info('Users:{}, Items:{}, Features:{} , Ratings:{}'
                     .format(user_count, item_count, feature_count, len(rating_ls)))

        self.user_mapping = core_utils.create_index(user_set)
        self.item_mapping = core_utils.create_index(item_set)
        self.feature_mapping = core_utils.create_index(feature_set)

        # Bias
        self.global_b = sum(rating_ls) / len(rating_ls)
        logging.info('Global Bias:{}'.format(self.global_b))
        self.bu = np.zeros(user_count)
        self.bi = np.zeros(item_count)

        # Matrix factorization
        logging.info("Factors:{}".format(self.mf_rank_rating))
        # self.P = np.random.rand(self.user_count, self.rank) * np.sqrt(2 / (self.user_count + self.item_count))
        # self.Q = np.random.rand(self.item_count, self.rank) * np.sqrt(2 / (self.user_count + self.item_count))
        self.P = np.random.normal(scale=1. / self.mf_rank_rating, size=(user_count, self.mf_rank_rating))
        self.Q = np.random.normal(scale=1. / self.mf_rank_rating, size=(item_count, self.mf_rank_rating))
        self.A = np.random.normal(scale=1. / self.mf_rank_rating, size=(feature_count, self.mf_rank_rating))
        # self.P = np.random.rand(user_count, self.mf_rank_rating) * np.sqrt(2 / (user_count + item_count))
        # self.Q = np.random.rand(item_count, self.mf_rank_rating) * np.sqrt(2 / (user_count + item_count))
        # self.A = np.random.rand(feature_count, self.mf_rank_rating) * np.sqrt(2 / (user_count + item_count))
        pass

    def sgd(self):
        for row in self.data:
            if len(row) > 2:
                user = row[self.data_index_user]
                i = self.user_mapping[user]

                item = row[self.data_index_item]
                j = self.item_mapping[item]

                rating = int(row[self.data_index_rating])

                predict = self.predict(i, j)

                e = rating - predict

                self.bu[i] += self.alpha * (e - self.beta * self.bu[i])
                self.bi[j] += self.alpha * (e - self.beta * self.bi[j])

                self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
                self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

                # Features

                # [
                #   ('purchase', 'good',     0.4754),
                #   ('purchase', 'very good',0.70)

                feature_pairs = row[self.data_index_features]
                if feature_pairs != '':
                    vector_total = np.zeros(self.mf_rank_rating)
                    list_features = eval(row[self.data_index_features])
                    for feature_data in list_features:
                        feature_name = feature_data[0]
                        diff = (self.P[i, :].dot(self.A[self.feature_mapping[feature_name]].T)) - float(feature_data[2])
                        vector_total += (self.A[self.feature_mapping[feature_name]] * diff)


                    self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * vector_total)
                    # self.P[i, :] += self.beta * vector_total

                    for feature_data in list_features:
                        feature_name = feature_data[0]
                        diff = (self.P[i, :].dot(self.A[self.feature_mapping[feature_name]].T)) - float(feature_data[2])
                        self.A[self.feature_mapping[feature_name]] += (self.beta * (self.P[i, :] * diff))

        pass

    def predict(self, i, j):
        return self.global_b + self.bu[i] + self.bi[j] + self.P[i, :].dot(self.Q[j, :].T)

    def predict_data(self, d):
        user = d[self.data_index_user]
        i = self.user_mapping[user]

        item = d[self.data_index_item]
        j = self.item_mapping[item]

        rating = int(d[self.data_index_rating])
        diff = rating - self.predict(i, j)
        return diff


if __name__ == "__main__":
    path = '../dataset/unit_test/my_set_all.dt'
    csv_reader = csv.reader(open(path), delimiter=',')
    data = [row for row in csv_reader if len(row) >= 2]
    data_info = {
        const.CONFIG_DATA_INDEX_USER: 1,
        const.CONFIG_DATA_INDEX_ITEM: 2,
        const.CONFIG_DATA_INDEX_RATING: 3,
        const.CONFIG_DATA_INDEX_FEATURES: 4,
        const.MF_RANK_RATING: 10,
        const.ITERATIONS: 100
    }
    model = AMF(data, data, data_info)
    model.train(20)
    pass
