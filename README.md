# Deeply Supervised Part Feature (DSPF) network
This project is for introducing how to use the code of the thesis of Deep Learning based Open-World Person Re-Identification.

# Requirements
* Cuda 9.0
* CuDnn 7.0
* Tensorflow 1.10
* Python 3.6

## Installation
You have first to install `Cuda 9.0` and `CuDnn 7.0` manually, which are required for gpu version of tensorflow.
For python packages we need, you can install them by:
```bash
> pip install -r requirements.txt
```

# Usage

## Dataset Preparation
We use `Market-1501` and `DukeMTMC-reID` datasets for performance evaluation. The datasets can be downloaded from [here](https://drive.google.com/file/d/1QpcheGYb_tbdmiDI8nvRTQoqc-E5pOvw/view). For open-world ReID settings, we follow the training/testing protocol proposed in this paper:
```
@INPROCEEDINGS{zheng2012,
author={W. {Zheng} and S. {Gong} and T. {Xiang}},
booktitle={2012 IEEE Conference on Computer Vision and Pattern Recognition},
title={Transfer re-identification: From person to set-based verification},
year={2012},
pages={2650-2657},
}
```
to partition each dataset into training/gallery/query set and store them in different folders. There is not the code or script for partitioning processes because some steps are works down by hand.

Next, the format of all images in each folder is transformed to .tfrecord file through the function `create_record` in `data/prepare_tfrecords.py`

Following the training/testing protocol in [zheng2012], you will get 10 groups of training/gallery/query set and the correspounding tfrecord file. For your each tfrecord file folder, it should like this:
```
├── tfrecord
   ├── train_market.tfrecords
   ├── gallery_market.tfrecords
   └── query_market.tfrecords
```

## Training
After finshing data preparation, you can start training the network by running `train.py`:
```bash
> python train.py --record_dir "the path of training tfrecord files" \ 
--dataset "market or duke" \ 
--device 0 \ 
--pre_model pretrain/se_resnext50/se_resnext50.ckpt \ 
--total_epochs 100 \ 
--learning_rate 0.001 \ 
--lamda 0.8 \ 
--data_aug \ 
--attention --deeplysupervised --part
```

The pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/1_kc-ikPhVzjgzWMrcSWMPH0IYTjrqOlY?usp=sharing), and then put them under this path: `./pretrain/se_resnext50/`.

In each training process, the network training needs to run `train.py` twice. In second time, you have to modify the setting of --pre_model into "the path of last model training in first time" and --learning_rate into 0.0001.

Because there are 10 groups of tfrecord files, the training process needs to repeat 10 times.

## Testing 
After training, the last models of each group will be used to extract features. 

You can convert all the gallery and query images in tfrecord files into the feature representataions with `FeatureExtraction.py`:
```bash
> python FeatureExtraction.py --record_dir "the path of gallery and query tfrecord files" \ 
--dataset "market or duke" \ 
--device 0 \ 
--feature_dir "the path of storing the output feature representataions"
--restore_model "the path of the last model training in second time" \
--iteration 10 \ 
--extract_gallery --extract_query \ 
--attention --deeplysupervised --part
```

Next, you can calculate TTR/FTR score with those extracted feature representataions through `evaluation.py`:

```bash
> python evaluation.py --record_dir --dataset "market or duke" \ 
--feature_dir "the path of stored feature representataions"
--result_dir "the path of the calculated TTR/FTR score" \
--iteration 10 \ 
--normalize_feat
```


# References
- [Se-ResNeXt-50 pretrained model](https://github.com/HiKapok/TF-SENet)
