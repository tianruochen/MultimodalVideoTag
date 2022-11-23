"""
feature reader
"""
#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import random
import os
import math
import json
import traceback
import pickle
python_ver = sys.version_info
from collections import defaultdict

import pandas as pd

from .ernie_task_reader import ExtractEmbeddingReader
from .reader_utils import DataReader

def preprocess_frame(video, max_frames):
    num_frames = video.shape[0]
    dim = video.shape[1]
    padding_length = max_frames - num_frames
    if padding_length > 0:
        fillarray = np.zeros((padding_length, dim))
        video_out = np.concatenate((video, fillarray), axis=0)
    else:
        video_out = video[:max_frames, ...]
    return video_out


def package_video_image_features(item, dim=1024):
    image_feature_paths = item["image_feature_paths"]
    video_feature_paths = item["video_feature_paths"]

    have_image_feature = len(image_feature_paths) > 0
    have_video_feature = len(video_feature_paths) > 0
    if not have_image_feature and not have_video_feature:
        image_video_features = np.zeros((1, dim))
        # image_video_features = preprocess_frame(image_video_features, 210)
        return image_video_features

    # 所有图像的特征封装为 30 * self.dim 的ndarray
    image_features = np.zeros((1, dim))
    if have_image_feature:
        image_features = np.zeros((30, dim))
        images_nums = len(image_feature_paths)
        seg_size = math.ceil(30 / images_nums)
        for i in range(images_nums):
            image_features[i:min(i + seg_size, 30), :] = np.load(image_feature_paths[i])

    videos_features = np.zeros((0, dim))
    if have_video_feature:
        for video_feature_path in video_feature_paths:
            if os.path.exists(video_feature_path):
                videos_features = np.load(video_feature_path)
                break

    # 如果有视频特征没有图像特征，则将视频第一帧特征作为图像特征
    if image_features.shape[0] == 1:
        image_features = np.zeros((30, dim))
        image_features[:] = videos_features[0]

    image_video_features = np.concatenate((image_features, videos_features), axis=0)
    # image_video_features = preprocess_frame(image_video_features, 210)

    return image_video_features


class ZyFeatureReader(DataReader):
    """
    Data reader for youtube-8M dataset, which was stored as features extracted by prior networks
    This is for the three models: lstm, attention cluster, nextvlad

    dataset cfg: num_classes
                 batch_size
                 list
                 NextVlad only: eigen_file
    """
    def __init__(self, name, mode, cfg):
        """
        init
        """
        self.name = name
        self.mode = mode
        self.num_classes = cfg.MODEL.num_classes

        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        # 训练或测试数据文件
        self.data_file = cfg[mode.upper()]["data_file"]
        # self.filelist = cfg[mode.upper()]['filelist']
        self.eigen_file = cfg.MODEL.get('eigen_file', None)
        self.num_seg = cfg.MODEL.get('num_seg', None)
        self.loss_type = cfg.TRAIN['loss_type']
        # 词表文件
        vocab_file = os.path.join(cfg.TRAIN.ernie_pretrain_dict_path,
                                  'vocab.txt')
        self.ernie_reader = ExtractEmbeddingReader(
            vocab_path=vocab_file,
            max_seq_len=cfg.MODEL.text_max_len,
            do_lower_case=True)
        # url_title_label_file = cfg[mode.upper()]['url_title_label_file']
        self.class_dict = load_class_file(cfg.MODEL.class_name_file)
        self.data = json.load(open(self.data_file))
        # self.url_title_info = load_video_file(url_title_label_file,
        #                                       self.class_dict, mode)

    def create_reader(self):
        """
        create reader
        """
        # url_list = list(self.url_title_info.keys())
        if self.mode == 'train':
            random.shuffle(self.data)

        def reader():
            """reader
            """
            batch_out = []
            for item in self.data:
                inv_id = item["pid"]
                try:
                    # rgb (210, 1024)
                    rgb = package_video_image_features(item).astype(float)

                    # audio (120, 1024)
                    audio = np.zeros((10, 128))
                    audio_feature_paths = item["audio_feature_paths"]
                    if audio_feature_paths and os.path.exists(audio_feature_paths[0]):
                        audio = np.load(audio_feature_paths[0])
                    # audio = preprocess_frame(audio, 120).astype(float)
                    audio_ = audio[:, :128]

                    text_raw = item['text_content']
                    text_one_hot = self.ernie_reader.data_generate_from_text(
                        str(text_raw))

                    # print(rgb.shape)
                    # print(audio.shape)
                    # print(text_one_hot.shape)
                    # print(text_raw)
                    if self.mode != 'infer':
                        label = item["labels_idx"]
                        label = [int(w) for w in label]
                        if self.loss_type == 'sigmoid':
                            label = make_one_hot(label, self.num_classes)
                        elif self.loss_type == 'softmax':
                            label = make_one_soft_hot(label, self.num_classes,
                                                      False)
                        batch_out.append((rgb, audio, text_one_hot, label))
                    else:
                        batch_out.append((rgb, audio, text_one_hot, inv_id))
                    if len(batch_out) == self.batch_size:
                        # print(batch_out)
                        yield batch_out
                        batch_out = []
                except Exception as e:
                    print("warning: load data {} failed, {}".format(
                        inv_id, str(e)))
                    traceback.print_exc()
                    continue


# if self.mode == 'infer' and len(batch_out) > 0:
            if len(batch_out) > 0:
                yield batch_out

        return reader

    def get_config_from_sec(self, sec, item, default=None):
        """get_config_from_sec
        """
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)


def load_video_file(label_file, class_dict, mode='train'):
    """
    labelfile formate: URL \t title \t label1,label2
    return dict
    """
    data = pd.read_csv(label_file, sep='\t', header=None)
    url_info_dict = defaultdict(dict)
    for index, row in data.iterrows():
        url = row[0]
        if url in url_info_dict:
            continue
        if pd.isna(row[1]):
            title = ""
        else:
            title = str(row[1])
        if mode == 'infer':
            url_info_dict[url] = {'title': title}
        else:
            if pd.isna(row[2]):
                continue
            labels = row[2].split(',')
            labels_idx = [class_dict[w] for w in labels if w in class_dict]
            if len(labels_idx) < 1:
                continue
            if url not in url_info_dict:
                url_info_dict[url] = {'label': labels_idx, 'title': title}
    print('load video %d' % (len(url_info_dict)))
    return url_info_dict


def dequantize(feat_vector, max_quantized_value=2., min_quantized_value=-2.):
    """
    Dequantize the feature from the byte format to the float format
    """

    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value

    return feat_vector * scalar + bias


epsilon = 0.1
smmoth_score = (1.0 / float(210)) * epsilon


def label_smmoth(label_one_hot_vector):
    """
    label_smmoth
    """
    global smmoth_score
    for i in range(len(label_one_hot_vector)):
        if label_one_hot_vector[i] == 0:
            label_one_hot_vector[i] = smmoth_score
    return label_one_hot_vector


def make_one_soft_hot(label, dim=15, label_smmoth=False):
    """
    make_one_soft_hot
    """
    one_hot_soft_label = np.zeros(dim)
    one_hot_soft_label = one_hot_soft_label.astype(float)
    # multi-labelis
    # label smmoth
    if label_smmoth:
        one_hot_soft_label = label_smmoth(one_hot_soft_label)
    label_len = len(label)
    prob = (1 - np.sum(one_hot_soft_label)) / float(label_len)
    for ind in label:
        one_hot_soft_label[ind] += prob
    #one_hot_soft_label = label_smmoth(one_hot_soft_label)
    return one_hot_soft_label


def make_one_hot(label, dim=15):
    """
    make_one_hot
    """
    one_hot_soft_label = np.zeros(dim)
    one_hot_soft_label = one_hot_soft_label.astype(float)
    for ind in label:
        one_hot_soft_label[ind] = 1
    return one_hot_soft_label


def generate_random_idx(feature_len, num_seg):
    """
    generate_random_idx
    """
    idxs = []
    stride = float(feature_len) / num_seg
    for i in range(num_seg):
        pos = (i + np.random.random()) * stride
        idxs.append(min(feature_len - 1, int(pos)))
    return idxs


def get_batch_ernie_input_feature(reader, texts):
    """
    get_batch_ernie_input_feature
    """
    result_list = reader.data_generate_from_texts(texts)
    result_trans = []
    for i in range(len(texts)):
        result_trans.append([result_list[0][i],\
                             result_list[1][i],
                             result_list[2][i],
                             result_list[3][i],
                             result_list[4][i]])
    return np.array(result_trans)


def load_class_file(class_file):
    """
    load_class_file
    """
    class_dict = {}
    if class_file.endswith("json"):
        tmp_class_dict = json.load(open(class_file))
        for class_name, idx in tmp_class_dict.items():
            class_dict[class_name] = int(idx)
    else:
        class_lines = open(class_file, 'r', encoding='utf8').readlines()
        for i, line in enumerate(class_lines):
            tmp = line.strip().split('\t')
            word = tmp[0]
            index = str(i)
            if len(tmp) == 2:
                index = tmp[1]
            class_dict[word] = index
    return class_dict


if __name__ == '__main__':
    pass