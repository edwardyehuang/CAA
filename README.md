# Channelized Axial Attention for Semantic Segmentation (AAAI-2022)

## News
Mar-16-2022: Based on the latest [Class-aware Regularizations (CAR)](https://github.com/edwardyehuang/CAR), CAA + ConvNeXt-Large achieved 64.12% mIOU on Pascal Context!

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/caa-channelized-axial-attention-for-semantic/semantic-segmentation-on-coco-stuff-test)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-stuff-test?p=caa-channelized-axial-attention-for-semantic)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/car-class-aware-regularizations-for-semantic-1/semantic-segmentation-on-pascal-context)](https://paperswithcode.com/sota/semantic-segmentation-on-pascal-context?p=car-class-aware-regularizations-for-semantic-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/caa-channelized-axial-attention-for-semantic/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=caa-channelized-axial-attention-for-semantic)

Some segmentation results on Flickr images:

<img src="https://raw.githubusercontent.com/edwardyehuang/iSeg/master/demo.png" width=450>

## Installation
1. Install TensorFlow (>= 2.4, 2.3 is not recommend for GPU, but okay for TPU)
2. Install iSeg (My personal segmentation codebase, update soon at https://github.com/edwardyehuang/iSeg)
3. Install iDS (Dataset supports for iSeg, update soon at https://github.com/edwardyehuang/iDS)
4. Clone this repo

## Model Zoo
Since some of the original experiments (especially for ResNet-101) are conducted a long time ago, the ckpts listed below may have slightly different performance with paper reported.

### Pascal Context

|  Backbone   | ckpts  | mIOU% | configs |
|  ----  | ----  | ---- | ---- |
| ResNet-101  | [weiyun](https://share.weiyun.com/nSUwp76n) |55.0|configs |
| EfficientNet-B7  | [weiyun](https://share.weiyun.com/uMXjsmXf)|60.3| configs |


### COCOStuff-10k

|  Backbone   | ckpts  | mIOU% | configs |
|  ----  | ----  | ---- | ---- |
| ResNet-101  | [weiyun](https://share.weiyun.com/LtcKwuhK) |41.2| configs |

### COCOStuff-164k

|  Backbone   | ckpts  | mIOU% | configs |
|  ----  | ----  | ---- | ---- |
| EfficientNet-B5  | [weiyun](https://share.weiyun.com/p5xbCE55) |47.27| configs |


## Inference on iPhone
Comming Soon

## Please cite us

```
@article{Huang2021CAAC,
  title={Channelized Axial Attention -- Considering Channel Relation within Spatial Attention for Semantic Segmentation},
  author={Y. Huang Di Kang and Wenjing Jia and X. He and Liu Liu},
  journal={ArXiv},
  year={2021},
  volume={abs/2101.07434}
}
```
