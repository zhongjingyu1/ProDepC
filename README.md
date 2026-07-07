## ProDepC--Prototype-Based and Dependency-Aware Calibrated Multi-Label Logit Adjustment for Long-Tailed Partial Multi-Label Classification

## Abstract 
Real-world multi-label classification (MLC) often suffers from two coupled challenges: (1) training instances are annotated with noisy candidate label sets in which only a subset of labels are correct, termed partial multi-label learning (PML); and (2) class frequencies follow a long-tailed (LT) distribution. Their combination makes tail classes extremely vulnerable to noise and leads to unstable learning, while candidate-label noise also distorts class-frequency estimates, undermining LT methods that rely on accurate priors. Moreover, conventional logit adjustment is typically derived under the single-label softmax competition assumption, whereas MLC models labels independently using sigmoid outputs; naively applying existing adjustments can limit gains for tail classes and introduce miscalibration. To systematically characterize and address this joint setting, we formalize the LT-PML problem and, from the perspective of balanced error rate optimization, derive a multi-label logit adjustment formulation (Multi-LA) that is compatible with independent-sigmoid multi-label modeling. Under the Hamming loss and a balanced pairwise ranking loss, we derive Bayes-optimal decision rules form and further incorporate label dependencies into the adjustment term, providing a principled basis for calibrated correction in MLC. Based on this theory, we propose a prototype-based and dependency-aware calibrated Multi-LA framework (ProDepC) for LT-PML. In ProDepC, class priors are robustly estimated online from prototype features, while the label-conditional dependency matrix is estimated from prediction-driven statistics. We then apply Multi-LA-based posterior calibration to improve candidate-label disambiguation, and design a Multi-LA-based asymmetric disambiguation loss that suppresses negative supervision while strengthening learning signals for tail classes. Extensive experiments demonstrate that ProDepC outperforms the strongest competitors by up to 9.06 percentage points in total mAP and 9.13 percentage points in tail mAP, while achieving 64.5\% mAP on the real-world NUS-WIDE dataset.

<img src='flow.png' width=1200>

## Requirements 
* [Pytorch](https://pytorch.org/)
* [Sklearn](https://scikit-learn.org/stable/)
## Dataset 
To evaluate/train Long-Tailed Partial Multi-Label Learning, first, the [VOC2012/2007](http://host.robots.ox.ac.uk/pascal/VOC/), [MSCOCO](https://cocodataset.org/#download) and [NUS-WIDE](https://huggingface.co/datasets/Lxyhaha/NUS-WIDE) datasets need to be downloaded. The image paths, labels, and captions for the VOC-MLT and COCO-MLT datasets can be found:

* [COCO-PMLT](https://github.com/zhongjingyu1/NDLP/tree/master/appendix)
* [VOC-PMLT](https://github.com/zhongjingyu1/NDLP/tree/master/appendix)
* [NW-PMLT](https://github.com/zhongjingyu1/NDLP/tree/master/appendix)
```
Shell
├── appendix
    ├── coco
        ├── coco_lt_train.txt
        ├── coco_lt_test.txt
        ├── coco_labels.txt
        ├── longtail2017
            ├── class_freq.pkl
            ├── class_split.pkl
    ├── VOCdevkit
        ├── voc_lt_train.txt
        ├── voc_lt_test.txt
        ├── voc_labels.txt
        ├── longtail2012
            ├── class_freq.pkl
            ├── class_split.pkl
    ├── nuswide
            ├── nw_lt_train.txt
            ├── nw_lt_test.txt
            ├── nw_labels.txt
            ├── class_freq.pkl
            ├── class_split.pkl
├── data
    ├── coco
        ├── train2017
            ├── 0000001.jpg
            ...
        ├── val2017
            ├── 0000002.jpg
            ...
    ├── voc
        ├── VOCdevkit
            ├── VOC2007
                ├── Annotations
                ├── ImageSets
                ├── JPEGImages
                    ├── 0000001.jpg
                    ...
                ├── SegementationClass
                ├── SegementationObject
            ├── VOC2012
                ├── Annotations
                ├── ImageSets
                ├── JPEGImages
                    ├── 0000002.jpg
                    ...
                ├── SegementationClass
                ├── SegementationObject
    ├── nuswide
        ├── images
            ├── 0_2124494179_b039ddccac_m.jpg
            ...
```
## Usage
### VOC-PMLT
```bash
python lmpt/train.py configs/voc/LT_resnet50_pfc_DB.py \
--dataset 'voc-lt' \
--seed '0' \
--batch_size 32 \
--epochs 10 \
--gamma 3
--partial_rate 0.3/0.5 \
--eta 0.9 \
--alpha_range 0.4,0.8 \
```
### COCO-PMLT or NW-PMLT
```bash
python lmpt/train.py configs/coco/LT_resnet50_pfc_DB.py \ 
--dataset 'voc-lt' \
--seed '0' \
--batch_size 32 \
--epochs 10 \
--gamma 2
--partial_rate 0.05/0.1 \
--eta 0.9 \
--alpha_range 0.4,0.8 \
```
## Pre-trained models
#### COCO-MLT
|   Backbone  | $$\rho$$  |    Total   |    Head   |  Medium  |   Tail  |      Download      |
| :---------: |:---------:| :------------: | :-----------: | :---------: | :---------: | :----------------: |
|  ResNet-50  |    0.05   |    40.06   |    41.87  |  40.57   | 37.80   | [model](https://drive.google.com/file/d/180zUlE_hNgtEEuLkFKXnqVUqYU1qLNkZ/view?usp=sharing)   |
|  ResNet-50  |    0.1    |    35.21   |    40.65  |  32.83   | 33.55   | [model](https://drive.google.com/file/d/1HGgrQ3qMM1ELeLQDsYGJUeDCaNAaAkPy/view?usp=sharing)   |

#### VOC-MLT
|   Backbone  | $$\rho$$  |    Total   |    Head   |  Medium  |   Tail  |      Download      |
| :---------: |:---------:| :------------: | :-----------: | :---------: | :---------: | :----------------: |
|  ResNet-50  |    0.3    |    59.51   |    59.54  |  70.33   | 51.38   | [model](https://drive.google.com/file/d/11xGmU5z7IT7_zmPELoMDTwwMGXNN2kbR/view?usp=sharing)   |
|  ResNet-50  |    0.5    |    40.76   |    43.09  |  52.59   | 30.13   | [model](https://drive.google.com/file/d/1vaYBkXGBduyt0AvEZMVVGm3z-7tuXeUH/view?usp=sharing)   |

#### NW-MLT
|   Backbone  | $$\rho$$  |    Total   |    Head   |  Medium  |   Tail  |      Download      |
| :---------: |:---------:| :------------: | :-----------: | :---------: | :---------: | :----------------: |
|  ResNet-50  |    0.05   |    29.43   |    33.04  |  23.31   | 51.78   | [model](https://drive.google.com/file/d/1beZI2ADMT2jQSyZZmdJ9oOt8ZM7Hev7h/view?usp=sharing)   |

## Acknowledgements
We use code from [DBL](https://github.com/wutong16/DistributionBalancedLoss) and [LMPT](https://github.com/richard-peng-xia/LMPT). We thank the authors for releasing their code.
## Contact
If you have any questions, please create an issue on this repository or contact at [23171214508@stu.xidian.edu.cn](mailto:23171214508@stu.xidian.edu.cn).



# ProDepC

<p align="center">
  <b>Prototype-Based and Dependency-Aware Calibrated Multi-Label Logit Adjustment<br>
  for Long-Tailed Partial Multi-Label Classification</b>
</p>

<p align="center">
  Jingyu Zhong · Ronghua Shang · Shasha Mao · Jie Feng
</p>

<p align="center">
  Official PyTorch implementation of <b>ProDepC</b>.
</p>

---

## Overview

Real-world multi-label data often exhibit two difficulties simultaneously: **candidate-label ambiguity** and **long-tailed class imbalance**. Their coexistence makes tail classes particularly vulnerable to noisy supervision and also makes class-prior estimation unreliable.

ProDepC addresses this setting through a **Multi-Label Logit Adjustment (Multi-LA)** framework derived for independent sigmoid-based multi-label modeling. The method combines:

- **Prototype-based class-prior estimation** for robust online estimation of marginal label frequencies;
- **Prediction-driven dependency estimation** for modeling label-conditional relationships without clean joint annotations;
- **Multi-LA-based posterior calibration** for more balanced candidate-label disambiguation;
- **Multi-LA-based asymmetric disambiguation (MLA-AD) loss** for suppressing unreliable negative supervision while strengthening tail-class learning.

<p align="center">
  <img src="assets/framework.png" width="100%" alt="Overview of Multi-LA and ProDepC">
</p>

The current code release focuses on the main **VOC-PMLT** and **COCO-PMLT** experiments.

---

## Highlights

- A Multi-LA formulation compatible with **independent sigmoid multi-label prediction**.
- Online class-prior estimation from a global prototype feature.
- Online label-dependency estimation from soft prediction statistics.
- Dynamic pseudo-label calibration for partial multi-label disambiguation.
- Lightweight training overhead: no extra network branch or additional pass over the training set.

---

## Repository Structure

```text
ProDepC/
├── assets/
│   └── framework.png              # Method overview
├── configs/
│   ├── voc/
│   │   └── LT_resnet50_pfc_DB.py
│   └── coco/
│       └── LT_resnet50_pfc_DB.py
├── data/
│   ├── voc/                       # VOC split metadata and LT statistics
│   └── coco/                      # COCO split metadata and LT statistics
├── mllt/                          # Model, dataset, loss, and evaluation infrastructure
├── pre-processed-data/            # Pre-generated partial-label files
├── utils/
│   └── PML_Confidence.py          # Core ProDepC implementation
├── create_longtail_dataset.py     # Long-tailed split construction utility
├── train_ours.py                  # Training entry point
├── test.py                        # Evaluation entry point
└── test_each_class.py             # Per-class evaluation utility
```

The main ProDepC logic is implemented in [`utils/PML_Confidence.py`](utils/PML_Confidence.py).

---

## Requirements

The codebase uses the MMCV 1.x training API and was developed with Python 3.8-era dependencies. A legacy PyTorch/MMCV environment is recommended.

### Core dependencies

- Python 3.8
- PyTorch and torchvision with CUDA support
- MMCV 1.x (`mmcv-full` for GPU execution)
- NumPy
- SciPy
- scikit-learn
- pandas
- Pillow
- tqdm
- matplotlib
- OpenCV
- pycocotools
- terminaltables

Additional modules imported by auxiliary code include `timm`, `torchmetrics`, `pynvml`, `lvis`, and `inplace_abn`. The main data loader also imports the OpenAI `clip` package; install it or remove the unused `import clip` line in `utils/imagelist.py`.

A typical environment can be created as follows:

```bash
conda create -n prodepc python=3.8 -y
conda activate prodepc

# Install a PyTorch/torchvision build compatible with your CUDA version first.
# Then install MMCV 1.x and the remaining dependencies.

pip install numpy scipy scikit-learn pandas pillow tqdm matplotlib \
            opencv-python pycocotools terminaltables timm torchmetrics pynvml lvis

# Required by the current utils/imagelist.py import
pip install git+https://github.com/openai/CLIP.git
```

> **Important:** MMCV 2.x is not directly compatible with the APIs used in this repository. Use an MMCV 1.x release that matches your PyTorch/CUDA environment.

---

## Dataset Preparation

### 1. VOC-PMLT

The training split is constructed from **Pascal VOC 2012**, while evaluation uses the **Pascal VOC 2007 test set**.

Expected structure:

```text
data/voc/
├── VOCdevkit/
│   ├── VOC2007/
│   │   ├── Annotations/
│   │   ├── ImageSets/
│   │   └── JPEGImages/
│   └── VOC2012/
│       ├── Annotations/
│       ├── ImageSets/
│       └── JPEGImages/
├── longtail2012/
├── voc_labels.txt
├── voc_lt_train.txt
└── voc_lt_test.txt
```

The repository already contains the long-tailed split metadata and pre-generated partial-label files for the reported noise settings.

Reported VOC-PMLT settings:

| Training images | Classes | Noise rates | Test images |
|---:|---:|:---:|---:|
| 1,142 | 20 | 0.1 / 0.3 / 0.5 | 4,952 |

### 2. COCO-PMLT

Place MS-COCO 2017 in the following structure:

```text
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
├── val2017/
├── longtail2017/
├── coco_labels.txt
├── coco_lt_train.txt
└── coco_lt_test.txt
```

Then edit the following line in [`configs/coco/LT_resnet50_pfc_DB.py`](configs/coco/LT_resnet50_pfc_DB.py):

```python
data_root = './data/coco/'
```

The original configuration contains a local absolute path and must be changed before running on another machine.

Reported COCO-PMLT settings:

| Training images | Classes | Noise rates | Test images |
|---:|---:|:---:|---:|
| 1,909 | 80 | 0.05 / 0.1 / 0.2 | 5,000 |

### Candidate-label files

During training, the code loads candidate-label files from `pre-processed-data/`, for example:

```text
voc-lt_0.1_sd0.npy
voc-lt_0.3_sd0.npy
voc-lt_0.5_sd0.npy
coco-lt_0.05_sd0.npy
coco-lt_0.1_sd0.npy
coco-lt_0.2_sd0.npy
```

If a requested file does not exist, `data/datasets.py` automatically generates it from the clean labels using the specified noise rate and seed.

---

## Training

Create the checkpoint directory first:

```bash
mkdir -p checkpoint
```

### VOC-PMLT

Example for noise rate `rho = 0.1`:

```bash
python train_ours.py \
  --config ./configs/voc/LT_resnet50_pfc_DB.py \
  --dataset voc-lt \
  --partial_rate 0.1 \
  --epochs 10 \
  --mu_range 0.9,0.99 \
  --m 0.9 \
  --phi 0.9 \
  --tau 2.0 \
  --beta 1.0 \
  --seed 0 \
  --gpus 1
```

For the other reported settings, change:

```bash
--partial_rate 0.3
--partial_rate 0.5
```

### COCO-PMLT

Example for noise rate `rho = 0.05`:

```bash
python train_ours.py \
  --config ./configs/coco/LT_resnet50_pfc_DB.py \
  --dataset coco-lt \
  --partial_rate 0.05 \
  --epochs 10 \
  --mu_range 0.99,0.999 \
  --m 0.99 \
  --phi 0.9 \
  --tau 2.0 \
  --beta 2.0 \
  --seed 0 \
  --gpus 1
```

For the other reported settings, change `--partial_rate` to `0.1` or `0.2`.

> **Code note:** the current `argparse` choices in `train_ours.py` omit `0.2`. To reproduce COCO-PMLT with `rho = 0.2`, add `0.2` to the `choices` list of `--partial_rate`.


### Recommended dataset-specific settings

The parameter analysis in the paper suggests the following settings for the main reported LT-PML results:

| Dataset | `mu_range` | `m` | `phi` | `beta` |
|:---|:---:|:---:|:---:|:---:|
| VOC-PMLT | `0.9,0.99` | `0.9` | `0.9` | `1.0` |
| COCO-PMLT | `0.99,0.999` | `0.99` | `0.9` | `2.0` |

The prior-adjustment scale `tau` performs best in a moderate range around the repository default; the example commands use `2.0`.

### Main hyperparameters

| Argument | Meaning | Default in code |
|:---|:---|:---:|
| `--mu_range` | Start/end momentum for pseudo-label updates | `0.9,0.99` |
| `--m` | Prototype feature momentum | `0.9` |
| `--phi` | Soft co-occurrence momentum | `0.9` |
| `--tau` | Class-prior adjustment strength | `2.0` |
| `--beta` | Label-dependency correction strength | `1.5` |
| `--epochs` | Number of training epochs | `10` |
| `--seed` | Random seed | `0` |

The backbone is ImageNet-pretrained **ResNet-50**. The implementation uses a 2048-to-256 projection layer and a 256-to-C classifier. Training uses batch size 32, 224x224 inputs, SGD with momentum 0.9 and weight decay `1e-4`, and linear warm-up during the first 500 iterations.

The asymmetric negative regularization scale is set internally to:

- `5` for VOC-PMLT;
- `2` for COCO-PMLT.

---

## Evaluation

Evaluate a saved checkpoint with:

```bash
python test.py \
  --config ./configs/voc/LT_resnet50_pfc_DB.py \
  --checkpoint ./checkpoint/your_checkpoint.pth \
  --eval multiple
```

For COCO-PMLT:

```bash
python test.py \
  --config ./configs/coco/LT_resnet50_pfc_DB.py \
  --checkpoint ./checkpoint/your_checkpoint.pth \
  --eval multiple
```

The evaluation script reports performance for the **Head**, **Middle**, **Tail**, and **Total** class groups, including mAP, accuracy, micro-F1, and macro-F1.

To print the standard mAP evaluation:

```bash
python test.py \
  --config ./configs/voc/LT_resnet50_pfc_DB.py \
  --checkpoint ./checkpoint/your_checkpoint.pth \
  --eval mAP
```

---

## Main Results

Total mAP (%) reported in the paper:

| Dataset | Noise rate | ProDepC-HL | ProDepC |
|:---|:---:|---:|---:|
| VOC-PMLT | 0.1 | 71.53 | **72.30** |
| VOC-PMLT | 0.3 | 60.60 | **61.49** |
| VOC-PMLT | 0.5 | 41.15 | **42.26** |
| COCO-PMLT | 0.05 | 42.83 | **43.89** |
| COCO-PMLT | 0.1 | 37.19 | **38.46** |
| COCO-PMLT | 0.2 | 31.62 | **32.87** |

Across the reported LT-PML benchmarks, ProDepC improves over the strongest competing methods by up to **9.06 percentage points in total mAP** and **9.13 percentage points in tail-class mAP**.

Results may vary slightly with hardware, CUDA/cuDNN versions, and random initialization. The paper reports five runs for the synthetic benchmark experiments.

---

## Implementation Notes

1. **The optimizer learning rate is controlled by the config file.**  
   Although `train_ours.py` defines a `--lr` argument, the current training code builds the optimizer from `cfg.optimizer`. The default learning rate in the provided VOC/COCO configs is `0.02`.

2. **COCO contains a machine-specific path in the released config.**  
   Update `data_root` before training.

3. **The provided long-tailed split metadata can be used directly.**  
   `create_longtail_dataset.py` is only needed when regenerating long-tailed splits; it contains local paths that should be adapted to your machine.

4. **This release is designed for a single-GPU workflow.**  
   `train_ours.py` uses `MMDataParallel` and the `--gpus` option controls the visible device list for non-distributed training.

---

## Citation

Please cite the paper when using this code. The final bibliographic information can be updated after publication.

```bibtex
@article{zhong2026prodepc,
  title   = {Prototype-Based and Dependency-Aware Calibrated Multi-Label Logit Adjustment for Long-Tailed Partial Multi-Label Classification},
  author  = {Zhong, Jingyu and Shang, Ronghua and Mao, Shasha and Feng, Jie},
  year    = {2026},
  note    = {Manuscript}
}
```

---

## Acknowledgements

This implementation uses components from the long-tailed multi-label learning infrastructure associated with **Distribution-Balanced Loss**, together with **MMCV**-style model, dataset, and evaluation utilities. We thank the authors of the related open-source projects.

---

## Contact

For questions about the paper or code, please open a GitHub issue.

