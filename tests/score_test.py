from acmf.core.feature_extraction import *

if __name__ == '__main__':

    print(tanh(0))

    print(tanh(score_cal((.5, .2))))
    print(tanh(score_cal((0, .2))))
    print(tanh(score_cal((.5, 0))))

