import pandas as pd
from collections import Counter
from acmf.core.feature_extraction import *
import acmf.core.nltkutils as nu
import acmf.core.feature_extraction as fe
import acmf.core.coreutils as core_utils

path = '../dataset/music_inst/Musical_Instruments_5_arr.json'
df = pd.read_json(path)

df['unixReviewTime'] = pd.to_datetime(df['unixReviewTime'], unit='s')
df['features'] = ""

df['context'] = core_utils.random_context(len(df), [3, 4])

df['feature_pair'] = fe.get_fe_pairs(df['reviewText'])

df[['reviewerID', 'asin', 'overall', 'feature_pair', 'context']].to_csv('Musical_Instruments_all.dt', header=False)
# df[['reviewText']].to_csv('Musical_Instruments_all_Text.dt', header=False)
# df[['feature_pair']].to_csv('Musical_Instruments_all_Features.dt', header=False)

# df[['feature_pair']].to_csv('Musical_Instruments_feature_pair.dt', header=False)

print("EXIT")
