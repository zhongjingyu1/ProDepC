import tqdm
import argparse
import warnings
from data.datasets import build_dataset1
from mllt.datasets import build_dataset
from utils.PML_Confidence import PML_Confidence
from utils.build_optimizer import build_optimizer
import os
import time
from utils.test_ephco import test_epoch
from utils.utils_algo import *
from mmcv import Config
from mllt.models import build_classifier
from mmcv.parallel import MMDataParallel
from mllt.datasets import build_dataloader
from collections import OrderedDict
from mllt.core.evaluation.mean_ap import eval_map
import torch.optim.lr_scheduler as lr_scheduler
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Noise Correction and Distribution Fine-Tuning', add_help=False)
    # data
    parser.add_argument('--config', default='./configs/voc/LT_resnet50_pfc_DB.py', choices=['./configs/voc/LT_resnet50_pfc_DB.py', './configs/coco/LT_resnet50_pfc_DB.py'], help='train config file path')
    parser.add_argument('--dataset', default='voc-lt', type=str, choices=['voc-lt', 'coco-lt'], help='dataset name')
    parser.add_argument('--data_dir', default='./pre-processed-data', type=str, help='experiment directory for loading pre-generated data')
    parser.add_argument('--partial_rate', default=0.1, type=float, choices=[0.05, 0.1, 0.3, 0.5], help='{COCO: 0.05, 0.1, 0.2} // {VOC: 0.1, 0.3, 0.5}')
    # method parameter
    parser.add_argument('--mu_range', default='0.9,0.99', type=str, help='warm-up ratio function parameter')
    parser.add_argument('--m', default=0.9, type=float, help='feature momentum updating parameter')
    parser.add_argument('--phi', default=0.9, type=float, help='co-occurrence updating parameter')
    parser.add_argument('--tau', default=2, type=float, help='Scaling parameter')
    parser.add_argument('--beta', default=1.5, type=float, help='Scaling label dependency parameter')
    # train
    parser.add_argument('--epochs', default=10, type=int, help='train epochs')
    parser.add_argument('--test_epochs', default=0, type=int, help='train epochs')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default='0', type=int, help='seed')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for optim')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use ''(only applicable to non-distributed training)')
    return parser

def main(args):
    print(args)
    cfg = Config.fromfile(args.config)
    torch.manual_seed(args.seed)

    def parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()
        return loss, log_vars

    def batch_processor(model, data):
        losses, output11, feat_x = model(**data)
        loss, log_vars = parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data),losses_all=losses)
        return outputs, output11, feat_x

    """ 
        model
    """
    if args.dataset=='coco-lt':
        args.num_class = 80
    elif args.dataset=='voc-lt':
        args.num_class = 20

    model = build_classifier(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.CLASSES = args.num_class
    model = MMDataParallel(model, device_ids=range(args.gpus)).cuda()
    model.train()

    """
    optimizer
    """
    optimizer = build_optimizer(model, cfg.optimizer)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max', verbose=True, min_lr=1e-7)
    """
    dataset and dataloader
    """
    train_data = build_dataset1(dataset=args.dataset, split='train', args=args)
    train_dataset = build_dataset(cfg.data.train)

    sampler_cfg = cfg.data.get('sampler_cfg', None)
    sampler = cfg.data.get('sampler', 'Group')

    train_dataset.targets = train_data.Partial_Labels
    train_dataset.targets_index = train_data.Partial_Labels_index
    data_loaders = build_dataloader(
        train_dataset,
        cfg.data.imgs_per_gpu,
        cfg.data.workers_per_gpu,
        args.gpus,
        dist=False,
        sampler=sampler,
        sampler_cfg=sampler_cfg)

    loss_fn = PML_Confidence(torch.from_numpy(train_data.Partial_Labels).squeeze(1).cuda(), args.dataset, args.mu_range, args.tau, args.beta, args.m, args.phi, args.epochs)

    """
    training
    """
    sf = nn.Softmax(dim=1)
    for epoch in range(args.epochs):
        model.train()
        gt_labels = []
        predict_p = []
        epoch_begin = time.time()
        for i, data_batch in enumerate(tqdm.tqdm(data_loaders)):
            labels = data_batch.pop('truelabel')
            index = data_batch.pop('index')
            partial_label = data_batch.get('gt_labels')
            inputs_datacenter = data_batch.get('img')
            inputs = inputs_datacenter.data[0]
            inputs, labels, partial_label, index = inputs.cuda(), labels.cuda(), partial_label.cuda().float(), index.cuda()
            optimizer.zero_grad()
            _, outputs, feature = batch_processor(model, data_batch)

            loss = loss_fn(outputs, feature, partial_label.float(), index, model, epoch)

            gt_labels.extend(labels.cpu().numpy().tolist())
            predict_p.extend(sf(outputs).cpu().detach().numpy())

            loss.backward()
            optimizer.step()
        t = time.time() - epoch_begin
        print("Epoch {} training ends, total {:.2f}s".format(i, t))
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        # 转换为MB并打印
        peak_mem_mb = peak_mem_bytes / (1024 ** 3)
        print(f"Epoch {i} GPU Memory Peak: {peak_mem_mb:.2f} GB")

        # 释放未使用的缓存
        torch.cuda.empty_cache()
        try:
            mAP, APs = eval_map(predict_p, gt_labels, None, print_summary=False)
            print("train epoch[{}/{}] loss:{:.3f} train mAP:{}".format(epoch + 1, args.epochs, loss, mAP))
        except:
            print('ValueError: Input contains NaN.')
            print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epochs, loss))
        if epoch > args.test_epochs:
            metrics_all = test_epoch(model, cfg)
            for split, mAP, micro_f1, macro_f1, acc in metrics_all:
                print('Split:{:>6s} mAP:{:.4f}  acc:{:.4f}  micro:{:.4f}  macro:{:.4f}'.format(
                    split, mAP, acc, micro_f1, macro_f1))
            current_mAP = metrics_all[3][1]
            exp_lr_scheduler.step(current_mAP)
            checkpoint_path = f'./checkpoint/{args.dataset}_{current_mAP}_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'checkpoint saved at: {checkpoint_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Noise Correction and Distribution Fine-Tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
