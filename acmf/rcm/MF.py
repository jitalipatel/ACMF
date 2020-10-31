import numpy as np
import logging

np.random.seed(123)

class MF:

    def __init__(self, data, rank, alpha, beta, itr_s):
        self.data = data
        self.num_users, self.num_items = self.data.shape
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.itr_s = itr_s

        # bias
        self.u = np.mean(self.data[np.where(self.data != 0)])
        self.bu = np.zeros(self.num_users)
        self.bi = np.zeros(self.num_items)

        # Matrix factorization
        self.P = np.random.normal(scale=1. / self.rank, size=(self.num_users, self.rank))
        self.Q = np.random.normal(scale=1. / self.rank, size=(self.num_items, self.rank))

        # create samples
        self.samples = [
            (i, j, self.data[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.data[i, j] > 0
        ]

    def train(self):
        for i in range(self.itr_s):
            self.sgd()
            print("Itr=", i, self.error())
        pass

    def error(self):
        error = 0
        for i, j, rating in self.samples:
            error += np.power(rating - self.predict(i, j), 2)
        return np.sqrt(error)
        pass

    def sgd(self):
        for i, j, rating in self.samples:
            prediction = self.predict(i, j)
            e = rating - prediction
            self.bu[i] += self.alpha * (e - self.beta * self.bu[i])
            self.bi[j] += self.alpha * (e - self.beta * self.bi[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])
            pass

    def predict(self, i, j):
        # return self.u + self.bu[i] + self.bi[j]
        return self.u + self.bu[i] + self.bi[j] + self.P[i, :].dot(self.Q[j, :].T)


if __name__ == "__main__":
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4]
    ])
    model = MF(R, 3, 0.01, 0.1, 20)
    model.train()
    # print(model.predict(0, 0))
