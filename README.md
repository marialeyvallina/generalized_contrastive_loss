# Generalized Contrastive Loss

Visual place recognition is a challenging task in computer vision and a key component of camera-based localization and navigation systems. Recently, Convolutional Neural Networks (CNNs) achieved high results and good generalization capabilities. They are usually trained using pairs or triplets of images labeled as either similar or dissimilar, in a binary fashion. In practice, the similarity between two images is not binary, but rather continuous. Furthermore, training these CNNs is computationally complex and involves costly pair and triplet mining strategies.
We propose a Generalized Contrastive loss (GCL) function that relies on image similarity as a continuous measure, and use it to train a siamese CNN. Furthermore, we propose three techniques for automatic annotation of image pairs with labels indicating their degree of similarity, and deploy them to re-annotate the MSLS, TB-Places, and 7Scenes datasets.
We demonstrate that siamese CNNs trained using the GCL function and the improved annotations consistently outperform their binary counterparts. Our models trained on MSLS outperform the state-of-the-art methods, including NetVLAD, and generalize well on the Pittsburgh, TokyoTM and Tokyo 24/7 datasets. Furthermore, training a siamese network using the GCL function does not require any pair mining. 

## Paper
If you use our code please cite our [paper](https://arxiv.org/abs/2103.06638)
```
@article{leyvavallina2021gcl,
  title={Generalized Contrastive Optimization of Siamese Networks for Place Recognition}, 
  author={María Leyva-Vallina and Nicola Strisciuglio and Nicolai Petkov},
  journal={arXiv preprint arXiv:2103.06638},
  year={2021}
  url={https://arxiv.org/abs/2103.06638}
}
```
## Network attention
![](https://github.com/marialeyvallina/generalized_contrastive_loss/blob/main/attention.png)

## Contact details
If you have any doubts please contact us at:
1. María Leyva-Vallina: m.leyva.vallina at rug dot nl
2. Nicola Strisciuglio: n.strisciuglio at utwente dot nl
## How to use this library
### Download the data
1. MSLS: The dataset is available on request [here](https://www.mapillary.com/dataset/places "MSLS"). For the new GT annotations, please register [here](https://forms.gle/zaG9vu8fCTT4FVcY6).
2. Pittsburgh: The whole dataset is available on request [here](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/ "Pittsburgh") and the train val splits for Pitts30k are available [here](https://www.di.ens.fr/willow/research/netvlad/ "Pitts30k"). 
3. TokyoTM: The dataset is available on request [here](https://www.di.ens.fr/willow/research/netvlad/ "TokyoTM"). 
4. Tokyo 24/7: The dataset is available on request [here](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/ "Tokyo 24/7"). 
5. TB-Places: The dataset is available [here](https://github.com/marialeyvallina/TB_Places "TB-Places"). For the new GT annotations, please register [here](https://forms.gle/zaG9vu8fCTT4FVcY6).
5. 7Scenes: The dataset is available [here](https://github.com/marialeyvallina/TB_Places "TB-Places"). For the new GT annotations, please register [here](https://forms.gle/zaG9vu8fCTT4FVcY6).
### Download the models
All our models can be downloaded from [here](https://drive.google.com/drive/folders/1RHxrAj062ZxDp5817t1s4OXGLP_i8JFX?usp=sharing).
### Our results
#### MSLS
| Backbone  | Whitening | Pooling | Dimensions | Loss |  R@1 |  R@5 | R@10 | mAP@1 | mAP@5 | mAP@10 |
|-----------|-----------|:-------:|------------|:----:|:----:|:----:|:----:|:-----:|:-----:|-----------------|
| VGG       | No        |   GeM<sup>[1](https://research.mapillary.com/img/publications/CVPR20c.pdf)</sup>  | 512        |  TL  |  28  |  35  |  49  |   -   |   -   | -               |
| VGG       | No        | NetVLAD<sup>[1](https://research.mapillary.com/img/publications/CVPR20c.pdf)</sup> | 32768      |  TL  |  30  |  40  |  44  |   -   |   -   | -               |
| VGG       | No        | NetVLAD<sup>[1](https://research.mapillary.com/img/publications/CVPR20c.pdf)</sup> | 32768      |  TL  |  48  |  58  |  64  |   -   |   -   | -               |
| VGG       | No        | PatchNetVLAD<sup>[2](https://arxiv.org/abs/2103.01486)</sup> | 32768      | TL   | 48.1 | 57.6  | 60.5 |   -   |   -   |        -        |
| ResNet50  | No        |   avg   | 2048       |  CL  | 24.9 | 39.0 | 44.6 |  24.9 |  16.8 | 14.8            |
| ResNet50  | No        |   avg   | 2048       |  GCL | 35.8 | 52.0 | 59.0 |  35.8 |  24.5 | 21.8            |
| ResNet50  | No        |   GeM   | 2048       |  CL  | 29.7 | 44.0 | 50.7 |  29.7 |  20.6 | 18.1            |
| ResNet50  | No        |   GeM   | 2048       |  GCL | 43.3 | 59.1 | 65.0 |  43.3 |   30  | 26.8            |
| ResNet152 | No        |   avg   | 2048       |  CL  | 29.7 | 44.2 | 51.3 |  29.7 |  19.4 | 17.2            |
| ResNet152 | No        |   avg   | 2048       |  GCL | 43.5 | 59.2 | 65.2 |  43.5 |  29.5 | 26.4            |
| ResNet152 | No        |   GeM   | 2048       |  CL  | 34.1 | 50.8 | 56.8 |  34.1 |  23.6 | 20.8            |
| ResNet152 | No        |   GeM   | 2048       |  GCL | 45.7 | 62.3 | 67.9 |  45.7 |  31.4 | 28.3            |
| ResNet50  | Yes       |   GeM   | 2048       |  GCL | 52.9 | 65.7 | 71.9 |  52.9 |  37.3 | 33.4            |
| ResNet152 | Yes       |   GeM   | 2048       |  GCL | 57.9 | 70.7 | 75.7 |  57.9 |  40.7 | 36.6          |
| ResNeXt-101-32x8d | Yes       |   GeM   | 1024       |  GCL | **62.3** | **76.2** | **81.1** |  **62.3** |  **47** | **43.8**           |

##### To reproduce them
Run the labeling/create_json_idx.py file to generate the necessary json index files for the dataset.

```shell
python3 labeling/create_json_idx.py --dataset msls --root_dir /mydir/MSLS/
```

Run the extract_predictions.py script to compute the map and query features, and the top-k prediction. For instance:
```shell
python3 extract_predictions.py --dataset MSLS --root_dir /mydir/MSLS/ --subset val --model_file models/MSLS/MSLS_resnet152_avg_480_GCL.pth --backbone resnet152 --pool avg --norm L2 --image_size 480,640 --batch_size 4
```
This will produce the file results/MSLS/val/MSLS_resnet152_avg_480_GCL_predictions.txt that you should use to evaluate the MSLS_resnet152_avg_480_GCL model in the [MSLS repository](https://github.com/mapillary/mapillary_sls/tree/master/mapillary_sls).


#### TB-Places
Run the extract_predictions.py script to compute the map and query features, and the map-query distances. For instance:
```shell
python3 extract_predictions.py --dataset TB_Places --root_dir /mydir/TB_Places/ --subset W18_W17 --model_file models/TB_Places/resnet34_avg_GCL.pth --backbone resnet34 --pool avg --image_size 224 --batch_size 4 --query_idx_file /mydir/TB_Places/W18/W18.json --map_idx_file /mydir/TB_Places/W17/W17.json --f_length 512
```
```shell
python3 extract_predictions.py --dataset TB_Places --root_dir /mydir/TB_Places/ --subset W18_map_query --model_file models/TB_Places/resnet34_avg_GCL.pth --backbone resnet34 --pool avg --image_size 224 --batch_size 4 --query_idx_file /mydir/TB_Places/W18/W18_query.json --map_idx_file /mydir/TB_Places/W18/W18_map.json --f_length 512
```
For obtaining the top-k recall, run the script eval_recallatk.py. By default, the K values are 1,2,3,4,5,10,15,20,25.
```shell
python3 eval_recallatk.py --prediction_distance_file results/TB_Places/W18_W17/resnet34_avg_GCL_distances.npy --gt_file /mydir/TB_Places/W18_W17_gt.h5 
```

```shell
python3 eval_recallatk.py --prediction_distance_file results/TB_Places/W18_map_query/resnet34_avg_GCL_distances.npy --gt_file /mydir/TB_Places/W18_map_query_gt.h5 
```
#### 7Scenes
Run the labeling/create_json_idx.py file to generate the necessary json index files for the dataset.

```shell
python3 labeling/create_json_idx.py --dataset 7scenes --root_dir /mydir/7Scenes/
```
Run the extract_predictions.py script to compute the map and query features, and the map-query distances. For instance:
```shell
python3 extract_predictions.py --dataset 7Scenes --root_dir /mydir/7Scenes/ --subset heads --model_file models/7Scenes/heads/resnet34_avg_GCL.pth --backbone resnet34 --pool avg --image_size 224 --batch_size 4 --query_idx_file /mydir/7Scenes/heads/test.json --map_idx_file /mydir/7Scenes/heads/train.json --f_length 512
```
This will produce the file results/7Scenes/heads/resnet34_avg_GCL_distances.npy, which we can use to evaluate the performance of the resnet34_avg_GCL model.

For obtaining the top-k recall, run the script eval_recallatk.py. By default, the K values are 1,2,3,4,5,10,15,20,25.
```shell
python3 eval_recallatk.py --prediction_distance_file results/7Scenes/heads/resnet34_avg_GCL_distances.npy --gt_file /mydir/7Scenes/heads_gt.h5 
```
For obtaining the Average Precision, run the script eval_recallatk.py.

```shell
python3 eval_AP.py --prediction_distance_file results/7Scenes/heads/resnet34_avg_GCL_distances.npy --gt_file /mydir/7Scenes/heads_gt.h5 
```
### Train your own models
Coming soon
### Define your own graded GT
Coming soon
