#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :analysis_results.py
# @Time     :2021/7/7 下午3:36
# @Author   :Chang Qing

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

from scenario_lib.metrics.gap_cal import calculate_gap

def get_best_recall_prec_f1(gd, pd):
    min_val, max_val = np.min(pd), np.max(pd)
    best_score = 0.
    best_thresh = 0.
    best_recall = 0.
    best_prec = 0.
    prec_ = 0
    recall_ = 0
    thresh_ = 0
    flag = 1
    for i in range(100):
        thresh = (max_val - min_val) / 100. * i + min_val - (max_val - min_val) / 100.
        cur_pd = pd.copy()
        cur_pd[cur_pd <= thresh] = 0
        cur_pd[cur_pd > thresh] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(gd, cur_pd)
        precision = precision[1]
        recall = recall[1]
        f1 = f1[1]

        if flag and precision >= 0.7:
            prec_ = precision
            recall_ = recall
            thresh_ = thresh
            flag = 0

        if (f1 > best_score):
            best_score = f1
            best_thresh = thresh
            best_recall = recall
            best_prec = precision
    # print("prec_70: {:.3f} recall: {:.3f} thresh: {:.3f}".format(prec_, recall_, thresh_))
    return best_recall, best_prec, best_score, best_thresh


def metrix_analysis(scores_matrix, labels_matrix, id2name, threshold=None, by_precision=True):
    assert scores_matrix.shape == labels_matrix.shape, f"Error, Shape not match: {scores_matrix.shape} != {labels_matrix.shape}"
    # for id, name in id2name.items():
    #     scores_for_one = scores_matrix[:, int(id)]
    #     labels_for_one = scores_matrix[:, int(id)]
    ytest = labels_matrix
    y_hat = scores_matrix
    print(f"ytest shape: {ytest.shape}")

    if not threshold:
        limit_threshs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    else:
        limit_threshs = [threshold]

    total_auc = 0.0
    total_f1 = 0.0
    aucs = []
    valid_cls_name = []
    nums = []
    # p_r_f1_th_list = [[[]] * 4] * len(recall_threshs)
    p_r_f1_th_list = [[[] for _ in range(4)] for th in limit_threshs]
    f1_s_best, prec_best, recs_best, thrs_best = [], [], [], []
    for i, name in id2name.items():
        # 排除不关注的类别
        i = int(i)
        if np.sum(ytest[:, i]) == 0:
            print("无效类别" + name)
            continue
        nums.append(np.sum(ytest[:, i]))  # 第i个类别的数量

        valid_cls_name.append(name)
        auc_for_i = roc_auc_score(ytest[:, i], y_hat[:, i])
        aucs.append(auc_for_i)

        # 每一个类别的p, r, th 序列
        # p-从小到大  r-从大到小  th-从小到大
        precs, recalls, threshes = precision_recall_curve(ytest[:, i], y_hat[:, i], pos_label=1)

        # print(len(precs), len(recalls), len(threshes))
        for k in range(len(limit_threshs)):
            if by_precision:
                for j in range(len(precs)):
                    if precs[j] >= limit_threshs[k]:
                        p_r_f1_th_list[k][0].append(precs[j])
                        p_r_f1_th_list[k][1].append(recalls[j])
                        p_r_f1_th_list[k][2].append(2 * precs[j] * recalls[j] / (precs[j] + recalls[j]))
                        p_r_f1_th_list[k][3].append(threshes[j - 1] if j - 1 > 0 else threshes[0])
                        # print(
                        #     "prec_{}: {:.3f} recall: {:.3f} thresh: {:.3f}".format(limit_threshs[k], precs[j],
                        #                                                            recalls[j],
                        #                                                            threshes[j - 1]))
                        break
            else:
                for j in range(len(precs)):
                    if recalls[j] < limit_threshs[k]:
                        p_r_f1_th_list[k][0].append(precs[j - 1] if j - 1 > 0 else precs[0])
                        p_r_f1_th_list[k][1].append(recalls[j - 1] if j - 1 > 0 else recalls[0])
                        p_r_f1_th_list[k][2].append(2 * precs[j - 1] * recalls[j - 1] / (precs[j - 1] + recalls[j - 1]))
                        p_r_f1_th_list[k][3].append(threshes[j - 2] if j - 2 > 0 else threshes[0])
                        # print(
                        #     "prec_{}: {:.3f} recall: {:.3f} thresh: {:.3f}".format(limit_threshs[k], precs[j - 1],
                        #                                                            recalls[j - 1],
                        #                                                            threshes[j - 2]))
                        break

        best_recall, best_prec, best_f1, best_thresh = get_best_recall_prec_f1(ytest[:, i], y_hat[:, i])
        prec_best.append(best_prec)
        recs_best.append(best_recall)
        thrs_best.append(best_thresh)
        f1_s_best.append(best_f1)

        print("auc:{:.3f}  best_recall:{:.3f}  best_prec:{:.3f}  best_f1:{:.3f}  best_thresh:{:.3f}  {:<20} ".format(
            auc_for_i, best_recall, best_prec, best_f1, best_thresh, name))

        total_auc += auc_for_i
        total_f1 += best_f1
    print(f"avg auc:{total_auc / len(id2name)}  avg f1: {total_f1 / len(id2name)}")

    # build csv
    save_dict = {"name": valid_cls_name,
                 "auc": aucs,
                 "nums": nums}
    # print(len(valid_cls_name))
    # print(len(aucs))
    # print(len(nums))

    if by_precision:
        # new_base_name = os.path.basename(to_save_path)[:-4] + "_limit_precision" + os.path.basename(to_save_path)[-4:]
        for k in range(len(limit_threshs)):
            save_dict[f"reca_@precision={limit_threshs[k]}"] = p_r_f1_th_list[k][1]
            save_dict[f"f1_s_@precision={limit_threshs[k]}"] = p_r_f1_th_list[k][2]
            save_dict[f"thre_@precision={limit_threshs[k]}"] = p_r_f1_th_list[k][3]
    else:
        # new_base_name = os.path.basename(to_save_path)[:-4] + "_limit_recall" + os.path.basename(to_save_path)[-4:]
        for k in range(len(limit_threshs)):
            save_dict[f"prec_@recall={limit_threshs[k]}"] = p_r_f1_th_list[k][0]
            save_dict[f"f1_s_@recall={limit_threshs[k]}"] = p_r_f1_th_list[k][2]
            save_dict[f"thre_@recall={limit_threshs[k]}"] = p_r_f1_th_list[k][3]
            # save_dict[f"precision_{recall_threshs[k]}"] = p_r_f1_th_list[k][0]

    save_dict["precison_bset"] = prec_best
    save_dict["recall_best"] = recs_best
    save_dict["thresh_best"] = thrs_best
    save_dict["f1_score_best"] = f1_s_best

    # to_save_path = os.path.join(os.path.dirname(to_save_path), new_base_name)
    #
    # df = pd.DataFrame(save_dict)
    # df.to_csv(to_save_path)
    return save_dict


def build_id2name_map(label2name_path):
    assert os.path.exists(label2name_path), f"{label2name_path} is not exists! Please check!"
    id2name = json.loads(open(label2name_path).read())
    return id2name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scores Matrix and Labels Matrix Analysis Script")
    parser.add_argument("--scores_file", type=str,
                        default="/home/work/changqing/MultimodalVideoTag/outputs/epoch13/AttentionLstmErnie_epoch13_acc76.1_scores_matrix_bd.npy",
                        help="the path of scores file (npy)")
    parser.add_argument("--labels_file", type=str,
                        default="/home/work/changqing/MultimodalVideoTag/outputs/epoch13/AttentionLstmErnie_epoch13_acc76.1_labels_matrix_bd.npy",
                        help="the path of labels file (npy)")
    parser.add_argument("--id2name_file", type=str,
                        default="/data02/changqing/ZyMultiModal_Data/annotations/v1_cls301/idx2name_cls301.json",
                        help="the path of id2name file (json)")
    parser.add_argument("--to_save_path", type=str,
                        default="../outputs/InsightMultiModal+AttentionLstmErnie_res_by_precision.csv",
                        help="to save path")
    args = parser.parse_args()

    scores_file = args.scores_file
    labels_file = args.labels_file
    id2name_file = args.id2name_file
    to_save_path = args.to_save_path

    scores_matrix = (np.load(scores_file) + np.load("/data02/changqing/InsightMultiModal_0825_104526_scores_matrix.npy")) / 2
    labels_matrix = np.load(labels_file)
    print(scores_matrix.shape)
    id2name = build_id2name_map(id2name_file)
    top_ks = [1, 3, 5]
    for top_k in top_ks:
        gap = calculate_gap(scores_matrix, labels_matrix, top_k=top_k)
        print(f"gap@top{top_k}: {gap}")

    save_dict = metrix_analysis(scores_matrix, labels_matrix, id2name, threshold=None, by_precision=True)

    df = pd.DataFrame(save_dict)
    df.to_csv(to_save_path)
    print("Analysis done!")

    # AttentionLstmErnie_epoch13_acc76.1:
    # multimodal_paddle: avg auc:0.963945214022243  avg f1: 0.4917679074839216
    # gap @ top1: 0.2959395441964685
    # gap @ top3: 0.5718484083989612
    # gap @ top5: 0.6440486855457146
