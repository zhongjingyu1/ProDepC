from cProfile import label
import os
import random
import numpy as np
import torch
from randaugment import RandAugment
import torchvision.transforms as transforms
from PIL import ImageDraw
from pipeline.handlers import VG256_handler
from utils.pml_data import generate_uniform_cv_candidate_labels
np.set_printoptions(suppress=True)

HANDLER_DICT = {'vg256': VG256_handler}

def get_datasets(args):

    train_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),CutoutPIL(cutout_factor=0.5),RandAugment(),transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),transforms.ToTensor()])

    # load data:
    source_data = load_data(args.path_images)
    data_handler = HANDLER_DICT[args.dataset]

    train_dataset = data_handler(source_data['train']['images'], source_data['train']['labels'], args.path_images, transform=train_transform)
    train_dataset.True_label = train_dataset.Y.copy()
    data_dir_prod = os.path.join('pre-processed-data')
    if not os.path.exists(data_dir_prod):
        os.makedirs(data_dir_prod)
    print('==> Loading local data copy in the partial multi-label setup')
    data_file = "{ds}_{pr}.npy".format(
        ds=args.dataset,
        pr=args.partial_rate)

    save_path = os.path.join(data_dir_prod, data_file)
    if not os.path.exists(save_path):
        Partial_Labels = generate_uniform_cv_candidate_labels(train_dataset.Y, args.partial_rate)
        data_dict = {
            'partial_labels': Partial_Labels
        }
        save_path = os.path.join(data_dir_prod, data_file)
        with open(save_path, 'wb') as f:
            np.save(f, data_dict)
        print('local data saved at ', save_path)
    else:
        data_dict = np.load(save_path, allow_pickle=True).item()
        Partial_Labels = data_dict['partial_labels']
    train_dataset.Y = Partial_Labels
    train_dataset.Partial = torch.from_numpy(Partial_Labels)
    val_dataset = data_handler(source_data['val']['images'], source_data['val']['labels'], args.path_images, transform=val_transform)

    return train_dataset, val_dataset



def load_data(base_path):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
    return data

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x