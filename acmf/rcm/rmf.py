import logging

import numpy as np

import acmf.core.constants as const
import acmf.core.coreutils as core_utils
from acmf.rcm.base_mf import BaseMF

np.random.seed(123)


class MF(BaseMF):
    """
    [[userId, ItemId, Rating],......]
    """

    def __init__(self, data, test, data_info, rank, epochs=10):
        self.user_id_index = data_info['user_id']
        self.item_id_index = data_info['item_id']
        self.rating_index = data_info['rating']
        self.data = data
        self.rank = rank
        self.epochs = epochs
        self.alpha = data_info[const.ALPHA]
        self.beta = data_info[const.BETA]
        self.test = test
        self.train_error = []
        self.test_error = []
        super().__init__(epochs, data, test)
        # Count USER, ITEMS
        self.user_set, self.item_set, self.rating_ls = set(), set(), []
        for d in data:
            if len(d) > 2:
                self.user_set.add(d[self.user_id_index])
                self.item_set.add(d[self.item_id_index])
                rating = int(d[self.rating_index])
                if rating != 0:
                    self.rating_ls.append(rating)
            else:
                logging.warn("Invalid:{}".format(d))

        # self.check_test()

        self.user_count, self.item_count = len(self.user_set), len(self.item_set)
        logging.info('Users:{}, Items:{}, Rating:{}'.format(self.user_count, self.item_count, len(self.rating_ls)))

        logging.info("{}".format(len(self.rating_ls) * 100 / (self.user_count * self.item_count)))

        # Bias
        self.global_b = sum(self.rating_ls) / len(self.rating_ls)
        logging.info("Global Bias:{}".format(self.global_b))

        self.bu = np.zeros(self.user_count)
        self.bi = np.zeros(self.item_count)

        self.user_index = core_utils.create_index(self.user_set)
        self.item_index = core_utils.create_index(self.item_set)

        # Matrix factorization
        self.P = np.random.normal(scale=1. / self.rank, size=(self.user_count, self.rank))
        self.Q = np.random.normal(scale=1. / self.rank, size=(self.item_count, self.rank))
        # self.P = np.random.rand(self.user_count, self.rank) * np.sqrt(2 / (self.user_count + self.item_count))
        # self.Q = np.random.rand(self.item_count, self.rank) * np.sqrt(2 / (self.user_count + self.item_count))
        logging.info("Matrix shapes: bu:{}, bi:{}, user_index:{}, item_index:{}, P:{}, Q:{}"
                     .format(len(self.bu), len(self.bi), len(self.user_index), len(self.item_index), self.P.shape,
                             self.Q.shape))

    def check_test(self):
        user_set, item_set = set(), set()
        for d in self.test:
            if len(d) > 2:
                user_set.add(d[self.user_id_index])
                item_set.add(d[self.item_id_index])
            else:
                logging.warn("Invalid:{}".format(d))

        logging.info("Test Len: {}, {}".format(len(user_set), len(item_set)))
        user_ls = [u for u in user_set if u in self.user_set]
        item_ls = [i for i in item_set if i in self.item_set]
        logging.info("{}, {}".format(len(user_ls), len(item_ls)))
        pass

    def sgd(self):
        for d in self.data:
            if len(d) > 2:
                user = d[self.user_id_index]
                i = self.user_index[user]

                item = d[self.item_id_index]
                j = self.item_index[item]

                rating = int(d[self.rating_index])

                predict = self.predict(i, j)
                e = rating - predict

                self.bu[i] += self.alpha * (e - self.beta * self.bu[i])
                self.bi[j] += self.alpha * (e - self.beta * self.bi[j])

                self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
                self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])
            else:
                logging.warn("Invalid:{}".format(d))
        pass

    def predict(self, i, j):
        return self.global_b + self.bu[i] + self.bi[j] + self.P[i, :].dot(self.Q[j, :].T)
        # return self.global_b + self.P[i, :].dot(self.Q[j, :].T)
        # return self.global_b + self.bu[i] + self.bi[j]

    def predict_data(self, d):
        user = d[self.user_id_index]
        i = self.user_index.get(user)
        if i is None:
            raise Exception("{} user not found in index".format(user))
        item = d[self.item_id_index]
        j = self.item_index.get(item)
        if j is None:
            raise Exception("{} item not found in index".format(item))
        rating = int(d[self.rating_index])
        diff = rating - self.predict(i, j)
        return diff

    # fig = plt.figure(figsize=(20, 10))
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # # ax1.set_title("P")
    # ax1.imshow(self.P, interpolation='nearest', aspect='auto')
    # ax2.imshow(self.Q, interpolation='nearest', aspect='auto')
    # plt.show()
