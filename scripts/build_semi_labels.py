#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :build_soft_labels.py
# @Time     :2022/8/31 下午3:12
# @Author   :Chang Qing

import sys

sys.path.append("../../")
import json
import numpy as np
from tqdm import tqdm
import pandas as pd


def save_to_json(save_target, save_path):
    json.dump(save_target, fp=open(save_path, "w"), indent=4, ensure_ascii=False)


trainval_data_path = "/data02/changqing/ZyMultiModal_Data/annotations/v2_cls301/trainval_cls301_101w.json"
trainval_data = json.load(open(trainval_data_path, "r"))
new_trainval_data_path = "/data02/changqing/ZyMultiModal_Data/annotations/v3_cls301/trainval_cls301_101w_semi.json"

trainval_output_path = "/data02/changqing/ZyMultiModal_Data/annotations/v3_cls301/AttentionLstmErnie_epoch13_acc76.1_trainval_outputs.json"
trainval_output = json.load(open(trainval_output_path, "r"))

pos_threshold = 0.93
neg_threshold = 0.01

new_trainval_data = []
print(len(trainval_data), len(trainval_output))

# assert len(trainval_data) == labels_matrix.shape[0] == scores_matrix.shape[0]
ign_nums = []
pid_list = []
labels_name_list = []
labels_idx_list = []
text_feature_path_list = []
video_feature_paths_list = []
audio_feature_paths_list = []
image_feature_paths_list = []
pos_labels_idx_list = []
neg_labels_idx_list = []
ign_labels_idx_list = []

for i, item in tqdm(enumerate(trainval_data), total=len(trainval_data)):
    pid = item["pid"]
    labels_name = item["labels_name"]
    labels_idx = item["labels_idx"]
    text_feature_path = item["text_feature_path"]
    video_feature_paths = item["video_feature_paths"]
    audio_feature_paths = item["audio_feature_paths"]
    image_feature_paths = item["image_feature_paths"]

    scores = trainval_output.get(pid, [])
    if not scores:
        print(pid)

    scores = np.array(scores)

    neg_labels_idx = list(map(int, np.where(scores < neg_threshold)[0]))
    pos_labels_idx = set(list(np.where(scores > pos_threshold)[0]) + labels_idx).difference(set(neg_labels_idx))
    pos_labels_idx = list(map(int, pos_labels_idx))
    ign_labels_idx = list(map(int, set(range(301)).difference(pos_labels_idx).difference(neg_labels_idx)))

    ign_num = 301 - len(pos_labels_idx) - len(neg_labels_idx)
    ign_nums.append(ign_num)

    pid_list.append(pid)
    labels_name_list.append(labels_name)
    labels_idx_list.append(labels_idx)
    text_feature_path_list.append(text_feature_path)
    video_feature_paths_list.append(video_feature_paths)
    audio_feature_paths_list.append(audio_feature_paths)
    image_feature_paths_list.append(image_feature_paths)
    pos_labels_idx_list.append(pos_labels_idx)
    neg_labels_idx_list.append(neg_labels_idx)
    ign_labels_idx_list.append(ign_labels_idx)

df = pd.DataFrame({
    "pids": pid_list,
    "labels_name": labels_name_list,
    "labels_idx": labels_idx_list,
    "text_feature_path_list": text_feature_path_list,
    "video_feature_paths_list": video_feature_paths_list,
    "audio_feature_paths_list": audio_feature_paths_list,
    "image_feature_paths_list": image_feature_paths_list,
    "pos_labels_idx_list": pos_labels_idx_list,
    "neg_labels_idx_list": neg_labels_idx_list,
    "ign_labels_idx_list": ign_labels_idx_list
})

print(len(new_trainval_data))
print(np.mean(ign_nums))
# save_to_json(new_trainval_data, new_trainval_data_path)
df.to_csv("/data02/changqing/ZyMultiModal_Data/annotations/v3_cls301/trainval_cls301_101w_semi.csv")