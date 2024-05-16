import pickle

pickle_path = "/root/autodl-tmp/features/Ocean/base_feature.pickle"

with open(pickle_path, 'rb') as f:
    data = pickle.load(f)
    dic = {'image_features_dict': data, 'semantic_features_dict': {}}


# path = "/root/autodl-tmp/features/Ocean/novel_features.plk"
# with open(path, 'wb') as f:
#     pickle.dump(dic, f)
#
#
# with open(path, 'rb') as f:
#     x = pickle.load(f)
#     print(x.keys())
