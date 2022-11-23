#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :build_trianval.py
# @Time     :2022/8/17 上午11:05
# @Author   :Chang Qing

import json

def save_to_json(save_target, save_path):
    json.dump(save_target, fp=open(save_path, "w"), indent=4, ensure_ascii=False)

def add_text_content(raw_data):
    new_data = []
    for item in raw_data:
        inv_id = item["pid"]
        text_content = id2text.get(inv_id, "")
        item["text_content"] = text_content
        new_data.append(item)
    return new_data


if __name__ == '__main__':
    id2text_path = "/data02/changqing/ZyMultiModal_Data/annotations/v1_cls301/id2text_content.json"
    id2text = json.load(open(id2text_path))
    # train_data_raw = json.load(open("/data02/changqing/ZyMultiModal_Data/annotations/v1_cls301/train_cls301.json", "r"))
    # valid_data_raw = json.load(open("/data02/changqing/ZyMultiModal_Data/annotations/v1_cls301/valid_cls301.json", "r"))
    #
    # trian_sample_nums = 10000
    # valid_sample_nums = 2000
    #
    # train_data = add_text_content(train_data_raw)
    # valid_data = add_text_content(valid_data_raw)
    #
    #
    # train_data_path = "../datasets/train.json"
    # valid_data_path = "../datasets/valid.json"
    #
    # train_samples_path = f"../datasets/train{trian_sample_nums}.json"
    # valid_samples_path = f"../datasets/valid{valid_sample_nums}.json"
    #
    # save_to_json(train_data, train_data_path)
    # save_to_json(valid_data, valid_data_path)
    # save_to_json(train_data[:trian_sample_nums], train_samples_path)
    # save_to_json(valid_data[:valid_sample_nums], valid_samples_path)

    test_data_raw = json.load(open("/data02/changqing/ZyMultiModal_TestData/annotations/v1_cls301/test_cls301.json", "r"))
    test_data = add_text_content(test_data_raw)
    test_data_path = "../datasets/test.json"
    save_to_json(test_data, test_data_path)

