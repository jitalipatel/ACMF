import logging
import unittest

import pandas as pd

import acmf.core.logconfig as conf
import acmf.core.nltkutils as nltk_utils


class NltkUtilTest(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        conf.set_log_config(logging.DEBUG, True)

    def test_pos_tagger(self):
        text = 'This is a Simple Program. It\'s written in Python'
        logging.info('Text: {0}'.format(text))
        out_ls = [
            [('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('Simple', 'JJ'), ('Program', 'NNP'), ('.', '.')],
            [('It', 'PRP'), ("'s", 'VBZ'), ('written', 'VBN'), ('in', 'IN'), ('Python', 'NNP')]
        ]
        self.assertEqual(out_ls, nltk_utils.get_pos_tags(text))

    def test_sentences(self):
        sents = nltk_utils.get_sentences(
            "Like another reviewer, I was going crazy comparing much more expensive covers - then found this.Figured that at this price point, if I get a year or two out of it, I'd be happy.It seems that it will last a whole lot longer though.It fits very well, and the bottom has a LOT of ventilation holes to keep things cool.I really like the prop-up &#34;feet&#34; on the bottom - makes it much easier for me to type at a desk, etc.The keyboard cover wasn't so great - fairly thick and rubbery and interfered with typing. I bought different different keyboard cover that's much better.")
        for s in sents:
            print()
        pass

    def test_count_opinions(self):

        lines = open('vader_lexicon.txt', encoding='utf-8').readlines()

        words = set()
        for l in lines:
            l = l.strip()
            if len(l) > 0:
                index = l.index('\t')
                word = l[0:index]
                words.add(word)

        path = '../dataset/unit_test/my_set_all_8.dt'
        # path = '../dataset/Electronics/out/merge_GE_USER_GE_50_all_1.dt'
        df = pd.read_csv(path, encoding='utf-8')
        ls = df['feature_pair'].tolist()
        ls = [eval(features) for features in ls if isinstance(features, str)]

        print('Reviews Total:', len(ls))

        opinions = []
        for inner_ls in ls:
            for sub_ls in inner_ls:
                if sub_ls[2] == 0.0:
                    opinions.append(sub_ls[1])

        print('Zero Total Pair :', len(opinions))

        count = 0
        for op in opinions:
            op_words = op.split(' ')
            not_found = [op_word for op_word in op_words if op_word not in words]
            if len(op_words) == len(not_found):
                count += 1
            else:
                print(op_words)

        print("All neutral:", count)

        pass


if __name__ == "__main__":
    unittest.main()
    pass
