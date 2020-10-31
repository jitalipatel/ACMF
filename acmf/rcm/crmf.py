import csv
import logging

import numpy as np

import acmf.core.constants as const
import acmf.core.coreutils as core_utils
from acmf.rcm.base_mf import BaseMF


class CMF(BaseMF):

    def __init__(self, data, test, info):
        self.data = data
        self.test = test
        self.user_data_index = info['user_data_index']
        self.item_data_index = info['item_data_index']
        self.rating_data_index = info['rating_data_index']
        self.context_data_index = info['context_data_index']
        self.iter = info['iter']
        self.mf_rank = info['mf_rank']
        self.context_info = info['context_info']
        self.alpha = info[const.ALPHA]
        self.beta = info[const.BETA]
        super().__init__(self.iter, data, test)
        # Count USER, ITEMS
        self.user_set, self.item_set, self.rating_ls = set(), set(), []
        for d in data:
            if len(d) > 2:
                self.user_set.add(d[self.user_data_index])
                self.item_set.add(d[self.item_data_index])
                rating = int(d[self.rating_data_index])
                if rating != 0:
                    self.rating_ls.append(rating)
            else:
                print(d)

        self.user_count, self.item_count = len(self.user_set), len(self.item_set)

        self.user_mapping = core_utils.create_index(self.user_set)
        self.item_mapping = core_utils.create_index(self.item_set)

        # Bias
        self.global_b = sum(self.rating_ls) / len(self.rating_ls)

        self.bu = [np.zeros(shape=(self.user_count, i)) for i in self.context_info]
        self.bi = [np.zeros(shape=(self.item_count, i)) for i in self.context_info]

        # Matrix factorization
        self.P = np.random.normal(scale=1. / self.mf_rank, size=(self.user_count, self.mf_rank))
        self.Q = np.random.normal(scale=1. / self.mf_rank, size=(self.item_count, self.mf_rank))
        # self.P = np.random.rand(self.user_count, self.mf_rank) * np.sqrt(2 / (self.user_count + self.item_count))
        # self.Q = np.random.rand(self.item_count, self.mf_rank) * np.sqrt(2 / (self.user_count + self.item_count))
        pass

    def train(self, inteval):
        for i in range(self.iter):
            self.sgd()
            if i % inteval == 0:
                logging.info(
                    "Epochs:{}, Error:{}".format(i, self.error(self.data)))
                    # "Epochs:{}, Train_Error:{}, Test_Error:{}".format(i, self.error(self.data), self.error(self.test)))
        logging.info(
            "Epochs:{}, Error:{}".format(i, self.error(self.data)))
            # "Train_Error:{}, Test_Error:{}".format(self.error(self.data), self.error(self.test)))

    def sgd(self):
        for row in self.data:
            i = self.user_mapping[row[self.user_data_index]]
            j = self.item_mapping[row[self.item_data_index]]
            rating = int(row[self.rating_data_index])

            context = eval(row[self.context_data_index])

            predict = self.predict(i, j, context)
            e = rating - predict

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

            arr_index = 0
            for c in context:
                self.bu[arr_index][i][c] += self.alpha * (e - self.beta * self.bu[arr_index][i][c])
                self.bi[arr_index][j][c] += self.alpha * (e - self.beta * self.bi[arr_index][j][c])
                arr_index += 1

        pass

    def predict(self, i, j, context):
        ans = self.global_b + self.P[i, :].dot(self.Q[j, :].T)
        arr_index = 0
        for c in context:
            ans += self.bu[arr_index][i][c] + self.bi[arr_index][j][c]
            arr_index += 1
        return ans

    def predict_data(self, d):
        i = self.user_mapping[d[self.user_data_index]]
        j = self.item_mapping[d[self.item_data_index]]
        rating = int(d[self.rating_data_index])
        context = eval(d[self.context_data_index])
        predict = self.predict(i, j, context)
        e = rating - predict
        return e


if __name__ == "__main__":
    path = '../dataset/unit_test/my_set_all.dt'
    csv_reader = csv.reader(open(path), delimiter=',')
    data = [row for row in csv_reader if len(row) >= 2]
    data_info = {
        "user_data_index": 1,
        "item_data_index": 2,
        "rating_data_index": 3,
        "context_data_index": 5,
        'iter': 100,
        'mf_rank': 10,
        'context_info': (3, 4)
    }
    model = CMF(data, data_info)
    model.train()
    model.plot()
    # model.predict("user", "item", (0, 0))
    pass
