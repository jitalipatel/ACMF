root_path = '/media/divyesh/WorkSpace/ML_Works/Projects/idea_projects/RecommenderSystem'

amazon_data_set_path = root_path + '/dataset/reviews_electronics_5'

amazon_data_set_original_file_path = amazon_data_set_path + "/Musical_Instruments_5.json"
amazon_data_set_original_array_file_path = amazon_data_set_path + "/Musical_Instruments_5_arr.json"

amazon_data_set_products_dir = amazon_data_set_path + "/products"

files = [
    '/4915_B007WTAJTO.json',
    '/501_B005LFT3GG.json'
]


def get_file_path():
    return amazon_data_set_products_dir + files[0]
