## ProDepC--Prototype-Based and Dependency-Aware Calibrated Multi-Label Logit Adjustment for Long-Tailed Partial Multi-Label Classification

## Abstract 
Real-world multi-label classification (MLC) often suffers from two coupled challenges: (1) training instances are annotated with noisy candidate label sets in which only a subset of labels are correct, termed partial multi-label learning (PML); and (2) class frequencies follow a long-tailed (LT) distribution. Their combination makes tail classes extremely vulnerable to noise and leads to unstable learning, while candidate-label noise also distorts class-frequency estimates, undermining LT methods that rely on accurate priors. Moreover, conventional logit adjustment is typically derived under the single-label softmax competition assumption, whereas MLC models labels independently using sigmoid outputs; naively applying existing adjustments can limit gains for tail classes and introduce miscalibration. To systematically characterize and address this joint setting, we formalize the LT-PML problem and, from the perspective of balanced error rate optimization, derive a multi-label logit adjustment formulation (Multi-LA) that is compatible with independent-sigmoid multi-label modeling. Under the Hamming loss and a balanced pairwise ranking loss, we derive Bayes-optimal decision rules form and further incorporate label dependencies into the adjustment term, providing a principled basis for calibrated correction in MLC. Based on this theory, we propose a prototype-based and dependency-aware calibrated Multi-LA framework (ProDepC) for LT-PML. In ProDepC, class priors are robustly estimated online from prototype features, while the label-conditional dependency matrix is estimated from prediction-driven statistics. We then apply Multi-LA-based posterior calibration to improve candidate-label disambiguation, and design a Multi-LA-based asymmetric disambiguation loss that suppresses negative supervision while strengthening learning signals for tail classes. Extensive experiments demonstrate that ProDepC outperforms the strongest competitors by up to 9.06 percentage points in total mAP and 9.13 percentage points in tail mAP, while achieving 64.5\% mAP on the real-world NUS-WIDE dataset.

<img src='./assets/FLOW.png' width=1200>

## :closed_book: Requirements 
* [Pytorch](https://pytorch.org/)
* [Sklearn](https://scikit-learn.org/stable/)
## :clipboard: Dataset 
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
## :page_with_curl: Usage
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
## :wrench: Pre-trained models
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

## :heartpulse: Acknowledgements
We use code from [DBL](https://github.com/wutong16/DistributionBalancedLoss) and [LMPT](https://github.com/richard-peng-xia/LMPT). We thank the authors for releasing their code.
## :mailbox_closed: Contact
If you have any questions, please create an issue on this repository or contact at [23171214508@stu.xidian.edu.cn](mailto:23171214508@stu.xidian.edu.cn).
## :pencil2: Citing
If you find this code useful, please consider to cite our work.
```
@article{zhong2026Noise,
  title={Noise correction and distribution fine-tuning for long-tailed partial multi-label learning},
  author = {Jingyu Zhong and Ronghua Shang and Mengmeng Liu and Yuhao Xie and Jie Feng},
  journal = {Pattern Recognition},
  volume = {180},
  pages = {114096},
  year = {2026},
}
```

