import json
import logging
import random
from os import listdir
from os.path import isfile, join

import pandas as pd


def convert_to_json_array(input_path, output_path):
    """
    This method is convert Simple JSON file to Array of Json Object
    Simply -> [data]
    :param input_path: input json file
    :param output_path: output json file
    """
    read_file = open(input_path, 'r')
    write_file = open(output_path, 'w')

    write_file.write('[')

    # read all lines
    lines = read_file.readlines()
    length = len(lines)
    logging.info('{} lines'.format(length))

    # Write upto second last record
    for index in range(length - 2):
        write_file.write(str(lines[index]).strip() + ",\n")

    # write last record with ] sign
    write_file.write(str(lines[length - 1]).strip() + "]")
    # closing files
    read_file.close()
    write_file.close()
    pass


def extract_products(input_path, out_dir):
    col_name = 'asin'
    data = json.load(open(input_path))
    logging.info("No of Objects:{}".format(len(data)))

    def get_pids():
        p_ids = set()
        for d in data:
            p_ids.add(d[col_name])
        logging.info("No of Sub-Objects:{}".format(len(p_ids)))
        return p_ids

    pids = get_pids()
    out_dic = {}
    for pid in pids:
        out_dic[pid] = list()

    for d in data:
        out_dic[d[col_name]].append(d)

    for pid, data in out_dic.items():
        l = len(data)
        if l >= 20 and l <= 50:
            logging.info("{}, {}".format(pid, l))
            out_file = open(out_dir + '/' + str(l) + '_' + str(pid) + '.json', 'w')
            json.dump(data, out_file)
            out_file.close()
    pass


# todo : Duplicated
def write_json_array(in_path, out_path):
    """
        Rewrite json file
        Format [ file_content ]
    :param in_path: input path
    :param out_path: output path
    :return:
    """
    first_char = '['
    last_char = ']'
    read_file = open(in_path, 'r')
    write_file = open(out_path, 'w')
    write_file.write(first_char)
    lines = read_file.readlines()
    length = len(lines)
    print("No of Lines:", length)
    for index in range(length - 2):
        write_file.write(str(lines[index]).strip() + ",\n")
    write_file.write(str(lines[length - 1]).strip() + last_char)


def json_dump():
    ls = [1, 2, (3, 4)]
    json.dump(ls, open("out.data", "w"))
    pass


def merge_products(mypath, out_path):
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    # print(str(onlyfiles))

    main_df = pd.DataFrame()
    for path in onlyfiles:
        df = pd.read_json(path)
        if len(df) >= 500:  # todo: TMP Condition
            print("Df:" + str(len(df)))
            main_df = main_df.append(df, ignore_index=True, sort=True)
            print("main:" + str(len(main_df)))
    main_df.to_csv(out_path)
    pass


def merge_products1(mypath, out_path, func):
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    logging.info("Files:{}".format(len(files)))

    files_filter = []
    total_size = 0
    for file_name in files:
        size = int(file_name[0:file_name.index('_')])
        if func(size):
            files_filter.append(file_name)
            total_size += size
        logging.info("{}, {}".format(total_size, len(files_filter)))

    main_df = pd.DataFrame()
    for file_name in files_filter:
        df = pd.read_json(join(mypath, file_name))
        main_df = main_df.append(df, ignore_index=True, sort=True)
        logging.info("{}, {}".format(file_name, len(main_df)))
    main_df.to_csv(out_path)
    pass


def write_csv(input_file, output_file):
    df = pd.read_json(input_file)
    df.to_csv(output_file, index=False)


def generate_movies_len(in_path, in_path_1, out_file_path):
    df = pd.read_csv(in_path, sep='::', names=['reviewerID', 'asin', 'overall', 'timeStamp'])
    df = df[['reviewerID', 'asin', 'overall']]

    length = len(df)
    rv_df = pd.read_csv(in_path_1, encoding='utf-8')
    rv_df = rv_df[0:length]

    df['feature_pair'] = rv_df['feature_pair']
    df['context'] = rv_df['context']
    df.to_csv(out_file_path)
    pass


def random_generate(file_path, out_file_path):
    users = 1000
    items = 100

    user_prefix = 'user_'
    item_prefix = 'item_'

    my_dict = {
        "reviewerID": [],
        "asin": [],
        "overall": [],
    }

    for user_id in range(0, users):
        per = random.randint(60, 75) / 100
        samples = int(items * per)
        list = random.sample(range(0, items - 1), samples)
        for item_id in list:
            rating = random.randint(1, 5)
            my_dict['reviewerID'].append(user_prefix + str(user_id))
            my_dict['asin'].append(item_prefix + str(item_id))
            my_dict['overall'].append(str(rating))

    len_dict = len(my_dict['reviewerID'])
    df = pd.read_csv(file_path, encoding='utf-8')
    df = df[0:len_dict]

    my_dict['feature_pair'] = df['feature_pair'].tolist()
    my_dict['context'] = df['context'].tolist()
    new_df = pd.DataFrame(my_dict)
    new_df.to_csv(out_file_path)
    pass
