import mmcv
import numpy as np
import torch
from mllt.datasets.dataset_wrappers import ConcatDataset
from mllt.core.evaluation.eval_tools import lists_to_arrays, eval_acc, eval_F1
from mllt.datasets import build_dataset
from mllt.datasets import build_dataloader
import argparse
import os
from mllt.core.evaluation.mean_ap import eval_map
from mmcv import Config
from mllt.models import build_classifier
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

def parse_args():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--config', default='./configs/coco/LT_resnet50_pfc_DB.py', help='train config file path')
    parser.add_argument('--checkpoint',default='./checkpoint/coco-lt_best_result.pth', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--eval', type=str, nargs='+', choices=['mAP', 'multiple'],default=['multiple'], help='eval metrics')
    parser.add_argument('--show', default=False, help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--testset_only', type=bool, default=True, help='only eval test set')
    parser.add_argument('--from_file', action='store_true', help='load network output results')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def make_dataset_list(cfg, test_only=True):
    cfg.data.test.test_mode = True
    test_dataset = build_dataset(cfg.data.test)
    dataset_list = [test_dataset]
    if test_only:
        return dataset_list

    if cfg.data.train.get('dataset', None) is not None:
        train_cfg = cfg.data.train.dataset
    else:
        train_cfg = cfg.data.train
    train_cfg.test_mode = True
    train_cfg.extra_aug = None
    train_cfg.flip_ratio = 0
    train_dataset = build_dataset(train_cfg)

    if isinstance(train_dataset, ConcatDataset):
        train_datasets = train_dataset.datasets
    else:
        train_datasets = [train_dataset]

    for train_dataset in train_datasets:
        dataset_list.append(train_dataset)

    return dataset_list


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset), bar_width=20)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result.cpu().numpy())
        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    model = build_classifier(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    dataset_list = make_dataset_list(cfg, True)
    savedata = [dict() for _ in range(len(dataset_list))]

    for d, dataset in enumerate(dataset_list):
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        gt_labels = []
        for i in range(len(dataset)):
            gt_ann = dataset.get_ann_info(i)
            gt_labels.append(gt_ann['labels'])

        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, False)

        savedata[d].update(gt_labels=gt_labels, outputs=np.vstack(outputs))

    img_prefixs = []
    dataset_list = make_dataset_list(cfg, False)
    img_ids = [[] for _ in range(len(dataset_list))]
    for d, dataset in enumerate(dataset_list):
        img_prefixs.append(dataset.img_prefix)
        img_infos = dataset.img_infos
        for i in range(len(dataset)):
            img_ids[d].append(img_infos[i]['id'])

    display_dict = {}
    eval_metrics = args.eval
    dataset = build_dataset(cfg.data.test)
    display_dict['class'] = dataset.CLASSES
    for i, data in enumerate(savedata):
        if args.testset_only and i > 0:  # test-set
            break
        gt_labels = data['gt_labels']
        outputs = data['outputs']

        gt_labels, outputs = lists_to_arrays([gt_labels, outputs])
        print('Starting evaluate {}'.format(' and '.join(eval_metrics)))
        for eval_metric in eval_metrics:
            if eval_metric == 'mAP':
                mAP, APs = eval_map(outputs, gt_labels, None, print_summary=True)
                display_dict['APs_{:1d}'.format(i)] = APs
            elif eval_metric == 'multiple':
                metrics = []
                for class_i in range(80):
                    selected_outputs = outputs[:, class_i]
                    selected_gt_labels = gt_labels[:, class_i]
                    classes = np.asarray(dataset.CLASSES)[class_i]
                    mAP, APs = eval_map(np.expand_dims(selected_outputs, axis=1), np.expand_dims(selected_gt_labels, axis=1), classes, print_summary=False)
                    metrics.append(mAP)
                import pandas as pd
                df = pd.DataFrame(metrics)
                df.to_csv('image_target.csv', index_label='ImageID')
if __name__ == '__main__':
    main()
