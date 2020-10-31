import datetime
import random

random.seed(123)

def cast_date(dates):
    """
    Convert list of String date to List of date
    :param dates: cast_date
    :return: list
    """
    return [datetime.datetime.fromtimestamp(d) for d in dates]


def random_context(length, context_list):
    out_list = []
    for l in range(length):
        out_list.append(tuple([random.randint(0, ls_context - 1) for ls_context in context_list]))
    return out_list


def random_context1(length, mod_int, context_list):
    out_list = []
    context = tuple([random.randint(0, ls_context - 1) for ls_context in context_list])
    for i in range(0, length):
        out_list.append(context)
        if i % mod_int == 0:
            context = tuple([random.randint(0, ls_context - 1) for ls_context in context_list])
    # random.shuffle(out_list)
    return out_list


def create_index(input_set):
    out_dict = {}
    c = 0
    for data in input_set:
        out_dict[data] = c
        c += 1
    return out_dict


def filter_features(data_list, vocab_path):
    ls = open(vocab_path, 'r').read().lower().split("\n")
    vocab = dict()
    for line in ls:
        parts = line.split(" ")
        if len(parts) == 2:
            value = parts[0]
            for key in parts[1].split(","):
                vocab[key] = value

    out_list = []
    for data in data_list:
        data = str(data)
        if len(data) == 0 or data == 'nan':
            out_list.append("")
        else:
            new_feature_ls = []
            for fo in eval(data):
                feature = fo[0]
                new_feature_str = ""
                for feature_key in feature.split(" "):
                    new_feature = vocab.get(feature_key)
                    if new_feature is None:
                        new_feature = feature_key
                    new_feature_str += " " + new_feature
                new_feature_str = new_feature_str.strip()
                new_feature_ls.append((new_feature_str, fo[1], fo[2]))
            out_list.append(new_feature_ls)
    pass
