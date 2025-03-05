# CaRaCount
This repository contains the official code and data for [Class-Agnostic Repetitive Action Counting Using Wearable Devices](https://ieeexplore.ieee.org/abstract/document/10452869) (TPAMI2025).


## Overview
Our work uses audio cues as exemplar to perform few-shot repetitive action counting in the wild on smartwatch data.

![Teaser](./assest/teaser.png)

## Get Started

Setup environment:
```bash
conda create -n adafocus python=3.9
conda activate adafocus
conda install pytorch=1.12.1 torchvision=0.13.1 -c pytorch
pip install numpy==1.26.0 tensorboardX
# if you are trying Uni-AdaFocus-X3D, run the following line
pip install iopath simplejson fvcore pytorchvideo psutil matplotlib opencv-python scipy pandas
```

## Reference
If you find our code or papers useful for your research, please cite:
```
@article{duc_caracount_tpami_2025,
     title = {Class-Agnostic Repetitive Action Counting Using Wearable Devices},
    author = {Duc Duy Nguyen, Lam Thanh Nguyen, Yifeng Huang, Cuong Pham, Minh Hoai},
   journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
      year = {2025},
}

@inproceedings{yifeng_exrac_aaai_2024,
     title = {Count What You Want: Exemplar Identification and Few-Shot Counting of Human Actions in the Wild},
    author = {Duc Duy Nguyen, Lam Thanh Nguyen, Yifeng Huang, Cuong Pham, Minh Hoai},
 booktitle = {Proceedings of AAAI Conference on Artificial Intelligence (AAAI)},
      year = {2024}
}
```

## Contact
If you have any question, feel free to contact the author. 

Duc Duy Nguyen: ducduy.nguyen@adelaide.edu.au
