"""
eval main
"""

import os
os.environ["FLAGS_eager_delete_tensor_gb"] = "0.0"
os.environ["FLAGS_sync_nccl_allreduce"] = "1"
os.environ["FLAGS_fast_eager_deletion_mode"] = "1"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0"
os.environ["FLAGS_memory_fraction_of_eager_deletion"] = "1"

import sys
sys.path.append("..")
import time
import argparse
import logging
import pickle

import numpy as np
import paddle.fluid as fluid
import paddle
paddle.enable_static()

from scenario_lib.accuracy_metrics import MetricsCalculator
from scenario_lib.datareader import get_reader
from scenario_lib.config import parse_config, merge_configs, print_configs
from scenario_lib.models.attention_lstm_ernie import AttentionLstmErnie
from scenario_lib.utils import test_with_pyreader


def parse_args():
    """parse_args
    """
    parser = argparse.ArgumentParser("Paddle Video evaluate script")
    parser.add_argument('--model_name',
                        type=str,
                        default='AttentionLstmErnie',
                        help='name of model to train.')
    parser.add_argument('--config',
                        type=str,
                        default='../conf/conf.txt',
                        help='path to config file of model')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help=
        'path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument('--output', type=str, default=None, help='output path')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=True,
                        help='default use gpu.')
    parser.add_argument('--save_model_param_dir',
                        type=str,
                        default="/home/work/changqing/MultimodalVideoTag/scenario_lib/checkpoints_train_86w/AttentionLstmErnie_epoch13_acc76.1",
                        help='checkpoint path')
    parser.add_argument('--save_inference_model',
                        type=str,
                        default="inference_models_save",
                        help='save inference path')
    parser.add_argument('--save_only',
                        action='store_true',
                        default=False,
                        help='only save model, do not evaluate model')
    args = parser.parse_args()
    return args


def evaluate(args):
    """evaluate
    """
    # parse config
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, 'Test')

    model_param_dir = args.save_model_param_dir
    basename = os.path.basename(model_param_dir)
    scores_matrix_path = f"/data02/changqing/ZyMultiModal_Data/annotations/v3_cls301/{basename}_trainval_scores_matrix_bd.npy"
    labels_matrix_path = f"/data02/changqing/ZyMultiModal_Data/annotations/v3_cls301/{basename}_trainval_labels_matrix_bd.npy"
    print(scores_matrix_path)
    print(labels_matrix_path)

    # build model
    test_model = AttentionLstmErnie(args.model_name,
                                     test_config,
                                     mode='test')
    startup = fluid.Program()
    test_prog = fluid.default_main_program().clone(for_test=True)
    with fluid.program_guard(test_prog, startup):
        with fluid.unique_name.guard():
            test_model.build_input(True)
            test_model.build_model()
            test_feeds = test_model.feeds()
            test_outputs = test_model.outputs()
            test_loss = test_model.loss()
            test_pyreader = test_model.pyreader()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)
    compiled_test_prog = fluid.compiler.CompiledProgram(test_prog)

    # load weights
    assert os.path.exists(args.save_model_param_dir), \
            "Given save weight dir {} not exist.".format(args.save_model_param_dir)
    test_model.load_test_weights_file(exe, args.save_model_param_dir,
                                       test_prog, place)

    if args.save_inference_model:
        save_model_params(exe, test_prog, test_model,
                          args.save_inference_model)

    if args.save_only is True:
        print('save model only, exit')
        return

    # get reader
    bs_denominator = 1

    test_config.TEST.batch_size = int(test_config.TEST.batch_size / bs_denominator)
    test_reader = get_reader(args.model_name.upper(), 'test', test_config)

    # get metrics
    test_metrics = MetricsCalculator(args.model_name.upper(), 'test',
                                      test_config)
    test_fetch_list = [test_loss.name] + [x.name for x in test_outputs
                                            ] + [test_feeds[-1].name]
    # get reader
    exe_places = fluid.cuda_places() if args.use_gpu else fluid.cpu_places()
    test_pyreader.decorate_sample_list_generator(test_reader,
                                                  places=exe_places)

    test_loss, metrics_dict_test = test_with_pyreader(exe, compiled_test_prog,
                                                      test_pyreader,
                                                      test_fetch_list,
                                                      test_metrics)

    scores_matrix = metrics_dict_test["pred_all"]
    labels_matrix = metrics_dict_test["label_all"]
    np.save(scores_matrix_path, scores_matrix)
    np.save(labels_matrix_path, labels_matrix)
    test_acc1 = metrics_dict_test['avg_acc1']
    print(test_loss)
    print(test_acc1)
    print("done")

def save_model_params(exe, program, model_object, save_dir):
    """save_model_params
    """
    feeded_var_names = [var.name for var in model_object.feeds()][:-1]
    fluid.io.save_inference_model(dirname=save_dir,
                                  feeded_var_names=feeded_var_names,
                                  main_program=program,
                                  target_vars=model_object.outputs(),
                                  executor=exe,
                                  model_filename='model',
                                  params_filename='params')

if __name__ == "__main__":
    args = parse_args()
    # 30w训练数据 [TEST] Finish	Loss: 97.205975,	avg_acc1:65.73,	avg_acc2:76.88,	avg_acc3:82.96,	avg_acc5:88.82,
    # 85w训练数据 [TEST] Finish	Loss: 77.033091,	avg_acc1:75.26,	avg_acc2:84.73,	avg_acc3:89.07,	avg_acc5:92.84,
    evaluate(args)