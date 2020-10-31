import numpy as np
import re
import acmf.core.coreutils as core_utils
import csv

np.random.seed(123)

class MF:
    """
    [[userId, ItemId, Rating],......]
    """

    def __init__(self, data, data_info, rank, itrs=10):
        self.user_id_index = data_info['user_id']
        self.item_id_index = data_info['item_id']
        self.rating_index = data_info['rating']
        self.data = data
        self.rank = rank
        self.itrs = itrs
        self.alpha = 0.01
        self.beta = 0.001

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
                print(d)

        self.user_count, self.item_count = len(self.user_set), len(self.item_set)

        # Bias
        self.global_b = sum(self.rating_ls) / len(self.rating_ls)
        self.bu = np.zeros(self.user_count)
        self.bi = np.zeros(self.item_count)

        self.user_index = core_utils.create_index(self.user_set)
        self.item_index = core_utils.create_index(self.item_set)

        # Matrix factorization
        self.P = np.random.normal(scale=1. / self.rank, size=(self.user_count, self.rank))
        self.Q = np.random.normal(scale=1. / self.rank, size=(self.item_count, self.rank))

        pass

    def train(self):
        for i in range(self.itrs):
            self.sgd()
            print(i, self.error())
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
        pass

    def predict(self, i, j):
        return self.global_b + self.bu[i] + self.bi[j] + self.P[i, :].dot(self.Q[j, :].T)
        # return self.global_b + self.bu[i] + self.bi[j]

    def error(self):
        error = 0
        total = 0
        for d in self.data:
            if len(d) > 2:
                user = d[self.user_id_index]
                i = self.user_index[user]

                item = d[self.item_id_index]
                j = self.item_index[item]

                rating = int(d[self.rating_index])
                error += np.power(rating - self.predict(i, j), 2)
                total += 1
        return np.sqrt(error/ total)
        pass


if __name__ == "__main__":
    path = '../dataset/unit_test/my_set_filters_last_1.dt'
    csv_reader = csv.reader(open(path), delimiter=',')
    data = [row for row in csv_reader if len(row) >= 2]
    data = data[1:]
    data_info = {
        "user_id": 1,
        "item_id": 2,
        "rating": 3
    }
    mf = MF(data, data_info, 15, 20)
    mf.train()
    pass
