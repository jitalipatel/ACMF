import logging

import numpy as np

import acmf.core.constants as const
import acmf.core.coreutils as core_utils
from acmf.rcm.base_mf import BaseMF


class ACRMF(BaseMF):

    def __init__(self, data, test, config):
        self.data = data
        self.test = test
        self.itrs = config[const.ITERATIONS]
        self.alpha = config[const.ALPHA]
        self.beta = config[const.BETA]
        logging.info("Data Size:{}".format(len(self.data)))

        self.data_index_user = config[const.CONFIG_DATA_INDEX_USER]
        self.data_index_item = config[const.CONFIG_DATA_INDEX_ITEM]
        self.data_index_rating = config[const.CONFIG_DATA_INDEX_RATING]
        self.data_index_context = config[const.CONFIG_DATA_INDEX_CONTEXT]
        self.data_context_info = config[const.CONFIG_CONTEXT_INFO]
        self.data_index_features = config[const.CONFIG_DATA_INDEX_FEATURES]
        self.mf_rank_rating = config[const.MF_RANK_RATING]
        super().__init__(self.itrs, data, test)

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
                print("Invalid Row:", row)

        user_count, item_count, feature_count = len(user_set), len(item_set), len(feature_set)

        logging.info('Users:{}, Items:{}, Features:{} , Ratings:{}'
                     .format(user_count, item_count, feature_count, len(rating_ls)))

        self.user_mapping = core_utils.create_index(user_set)
        self.item_mapping = core_utils.create_index(item_set)
        self.feature_mapping = core_utils.create_index(feature_set)

        # Bias
        self.global_b = sum(rating_ls) / len(rating_ls)
        logging.info('Global Bias:{}'.format(self.global_b))

        self.bu = [np.zeros(shape=(user_count, i)) for i in self.data_context_info]
        self.bi = [np.zeros(shape=(item_count, i)) for i in self.data_context_info]

        for arr in self.bu:
            logging.info(arr.shape)
        for arr in self.bi:
            logging.info(arr.shape)

        # Matrix factorization
        logging.info("Factors:{}".format(self.mf_rank_rating))
        self.P = np.random.normal(scale=1. / self.mf_rank_rating, size=(user_count, self.mf_rank_rating))
        self.Q = np.random.normal(scale=1. / self.mf_rank_rating, size=(item_count, self.mf_rank_rating))
        self.A = np.random.normal(scale=1. / self.mf_rank_rating, size=(feature_count, self.mf_rank_rating))

        # self.P = np.random.rand(user_count, self.mf_rank_rating) * np.sqrt(2 / (user_count + item_count))
        # self.Q = np.random.rand(item_count, self.mf_rank_rating) * np.sqrt(2 / (user_count + item_count))
        # self.A = np.random.rand(feature_count, self.mf_rank_rating) * np.sqrt(2 / (user_count + item_count))
        logging.info("{}, {}, {}".format(self.P.shape, self.Q.shape, self.A.shape))
        pass

    def train(self, interval):
        for i in range(self.itrs):
            self.sgd()
            if i % interval == 0:
                logging.info(
                    "Epochs:{}, Error:{}".format(i, self.error(self.data)))
                    # "Epochs:{}, Train_Error:{}, Test_Error:{}".format(i, self.error(self.data), self.error(self.test)))
        logging.info(
            "Epochs:{}, Error:{}".format(i, self.error(self.data)))
            # "Train_Error:{}, Test_Error:{}".format(self.error(self.data), self.error(self.test)))
        pass

    def sgd(self):
        for row in self.data:
            if len(row) > 2:

                user = row[self.data_index_user]
                i = self.user_mapping[user]

                item = row[self.data_index_item]
                j = self.item_mapping[item]

                rating = int(row[self.data_index_rating])

                context = eval(row[self.data_index_context])

                predict = self.predict(i, j, context)
                e = rating - predict

                self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
                self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

                arr_index = 0
                for c in context:
                    self.bu[arr_index][i][c] += self.alpha * (e - self.beta * self.bu[arr_index][i][c])
                    self.bi[arr_index][j][c] += self.alpha * (e - self.beta * self.bi[arr_index][j][c])
                    arr_index += 1

                # Features
                feature_pairs = row[self.data_index_features]
                if feature_pairs != '':
                    vector_total = np.zeros(self.mf_rank_rating)
                    list_features = eval(row[self.data_index_features])
                    for feature_data in list_features:
                        feature_name = feature_data[0]
                        diff = (self.P[i, :].dot(self.A[self.feature_mapping[feature_name]].T)) - float(feature_data[2])
                        vector_total += (self.A[self.feature_mapping[feature_name]] * diff)

                    self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * vector_total)

                    for feature_data in list_features:
                        feature_name = feature_data[0]
                        diff = (self.P[i, :].dot(self.A[self.feature_mapping[feature_name]].T)) - float(feature_data[2])
                        self.A[self.feature_mapping[feature_name]] += (self.beta * (self.P[i, :] * diff))

        pass

    def predict(self, i, j, context):
        ans = self.global_b + self.P[i, :].dot(self.Q[j, :].T)
        arr_index = 0
        for c in context:
            ans += self.bu[arr_index][i][c] + self.bi[arr_index][j][c]
            arr_index += 1
        return ans

    def predict_data(self, d):
        user = d[self.data_index_user]
        i = self.user_mapping[user]

        item = d[self.data_index_item]
        j = self.item_mapping[item]

        rating = int(d[self.data_index_rating])

        context = eval(d[self.data_index_context])

        predict = self.predict(i, j, context)
        e = rating - predict
        return e
