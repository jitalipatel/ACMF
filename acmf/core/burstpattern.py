import pandas as pd
import logging


class BurstPattern:
    """
    Rule based Pattern capturing
    Initially:
        rule 1: i > i+1
        rule 2: i-1 < i
        rule 3: i-1 < i > i+1
    """

    def __init__(self, df, splitter=('1Y', '1M')):
        """
        :param df: dataFrame
        :param splitter: splitter
        """
        logging.info("-------- BURST PATTERN ----------")
        self.out_dict = {}
        self.df = df
        self.splitter = splitter
        self.total_count = df['count'].sum()
        logging.info("DatSet:{}, splitter:{}".format(len(self.df), splitter))
        logging.info("Total Count:{}".format(self.total_count))
        logging.debug("------------------------------")
        pass

    def match(self):
        """
        Pattern matching process
        :return: dict
        """
        logging.info("Matching Process......")
        grp_list = [(self.df, 0)]
        while len(grp_list) > 0:
            # pick first frame from list
            my_df, my_index = grp_list[0]
            my_name = self.splitter[my_index]

            logging.debug("DataSet:{}, Splitter:{}".format(len(my_df), my_name))

            # group by Date
            my_grp = my_df.groupby(pd.Grouper(key='date', freq=my_name))
            group_data = [grp_data for grp_name, grp_data in my_grp]
            group_data_count = [d['count'].sum() for d in group_data]
            group_data_len = [len(d) for d in group_data]

            sum_data = sum(group_data_count)
            total_windows_k = len(my_grp)
            avg_reviews = int(sum_data / total_windows_k)

            logging.info("Total Windows:{} = {}".format(group_data_len, total_windows_k))
            logging.info("Total Count:{}".format(group_data_count))
            logging.info("Sum:{}, Avg:{}".format(sum_data, avg_reviews))

            tuples = BurstPattern.__is_match(group_data_count, avg_reviews)
            if len(tuples) > 0:
                # Add Entry
                logging.debug("Match found (index, pattern_id) :{}".format(tuples))
                self.__update_indexes(group_data, tuples, self.splitter[my_index])
                # Prepare Next Iteration
                if (my_index + 1) < len(self.splitter):
                    for t in tuples:
                        grp_list.append((group_data[t[0]], my_index + 1))
            # Remove current group
            grp_list.remove(grp_list[0])
            logging.debug("------------------------------")

        logging.info("Total indexes:{}".format(len(self.out_dict)))
        return self.out_dict

    @staticmethod
    def __is_match(count_list, avg_reviews):
        """
        Pattern identification
        :param count_list: windows count
        :param avg_reviews: average
        :return: [(index, pattern_id), ...]
        """
        return_list = []
        max_len = len(count_list)
        for i in range(max_len):
            if count_list[i] > avg_reviews:
                # For first element
                if i == 0 and i + 1 < max_len and count_list[i] > count_list[i + 1]:
                    return_list.append((i, 1))
                # For all middle element
                elif i - 1 >= 0 and count_list[i - 1] < count_list[i] and i + 1 < max_len and count_list[i] > \
                        count_list[i + 1]:
                    return_list.append((i, 2))
                # For last element
                elif max_len - 1 == i and i - 1 >= 0 and count_list[i] > count_list[i - 1]:
                    return_list.append((i, 3))
        return return_list
        pass

    def __update_indexes(self, group_data, tuples, splitter):
        """
        Update indexes : Remove parent entry and add new child entry
        Like : 1M -> 1W, So remove 1M and add 1W
        :param group_data: dataFrame
        :param tuples: match pattern
        :param splitter: used splitter
        :return: None
        """
        # remove parent entry
        for data in group_data:
            for i in data.index.values:
                self.out_dict.pop(i, None)

        # add child entry
        for t in tuples:
            for d in [group_data[t[0]]]:
                for i in d.index.values:
                    self.add_entry(i, (splitter, t[1]))

        logging.info("Total indexes:{}".format(len(self.out_dict)))

    def add_entry(self, index, tp):
        """
        add entry by index
        :param index: index
        :param tp: pattern information
        :return: none
        """
        ls = self.out_dict.get(index)
        if ls is None:
            ls = []
        ls.append(tp)
        self.out_dict[index] = ls
