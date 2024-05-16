import os
import numpy as np
from gensim.models import KeyedVectors

current_path = os.path.dirname(__file__)
wv_model = KeyedVectors.load_word2vec_format("/root/autodl-tmp/GoogleNews-vectors-negative300.bin", binary=True)


def getLabel(isBase):
    labels = []
    data_dir = "/root/autodl-tmp/dataSet/Ocean/images"
    if isBase:
        for cls in os.listdir(data_dir)[:64]:
            labels.append(cls.replace(' ', '_'))
    else:
        for cls in os.listdir(data_dir)[80:]:
            labels.append(cls.replace(' ', '_'))
    return labels


def extract_word_vec(image_labels):

    # tmp_wv = np.zeros(300)
    # c = 0
    # for word in words:
    #     if word.lower() in wv_model.key_to_index.keys():
    #         tmp_wv += wv_model[word.lower()]
    #         c += 1
    #     else:
    #         print('{} in {} does not exist in w2v models'.format(word, words))
    # if c != 0:
    #     return tmp_wv / c
    # else:
    #     print('No embedding for :', words)
    #     return tmp_wv

    # 提取每个标签的词向量
    label_vectors =[]
    for label in image_labels:
        if label.lower() in wv_model.key_to_index.keys():
            label_vectors.append(wv_model[label.lower()])
        elif len(label.lower.split('_')) > 1:
            flag = False
            for sub_label in label.lower.split('_'):
                if sub_label in wv_model.key_to_index.keys():
                    label_vectors.append(wv_model[sub_label])
                    flag = True
                    break
            if flag:
                print(f"{label.lower}不在模型库中！！！！")
                label_vectors.append([0 for i in range(300)])

        else:
            print(f"{label.lower}不在模型库中！！！！")
            label_vectors.append([0 for i in range(300)])

    np_v = np.asarray(label_vectors)
    print(np_v.shape)
    # 打印词向量
    for label, vector in zip(image_labels, label_vectors):
        print(f"Label: {label}, Vector: {vector}")


if __name__ == "__main__":
    image_labels = getLabel(isBase=False)
    extract_word_vec(image_labels)
    # print(wv_model["anemone"])
    # print(wv_model["anemone"][0])


