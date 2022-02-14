# Generalized Contrastive Loss
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-contrastive-optimization-of/visual-place-recognition-on-mapillary-test)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-test?p=generalized-contrastive-optimization-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-contrastive-optimization-of/visual-place-recognition-on-mapillary-val)](https://paperswithcode.com/sota/visual-place-recognition-on-mapillary-val?p=generalized-contrastive-optimization-of)

Visual place recognition is a challenging task in computer vision and a key component of camera-based localization and navigation systems. Recently, Convolutional Neural Networks (CNNs) achieved high results and good generalization capabilities. They are usually trained using pairs or triplets of images labeled as either similar or dissimilar, in a binary fashion. In practice, the similarity between two images is not binary, but rather continuous. Furthermore, training these CNNs is computationally complex and involves costly pair and triplet mining strategies.
We propose a Generalized Contrastive loss (GCL) function that relies on image similarity as a continuous measure, and use it to train a siamese CNN. Furthermore, we propose three techniques for automatic annotation of image pairs with labels indicating their degree of similarity, and deploy them to re-annotate the MSLS, TB-Places, and 7Scenes datasets.
We demonstrate that siamese CNNs trained using the GCL function and the improved annotations consistently outperform their binary counterparts. Our models trained on MSLS outperform the state-of-the-art methods, including NetVLAD, and generalize well on the Pittsburgh, TokyoTM and Tokyo 24/7 datasets. Furthermore, training a siamese network using the GCL function does not require any pair mining. 

## Paper and license
The code is licensed under the [MIT License](license.md).

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

| ** **                 |           |         |          | **MSLS-Val** |          |          | **MSLS-Test** |          |          | **Pitts30k** |          |          | **Tokyo24/7** |          |               | **RobotSeasons v2- all** |              |               | **Extended CMU-all** |              |
| **Method**            | **PCA_w** | **Dim** | **R@1**  | **R@5**      | **R@10** | **R@1**  | **R@5**       | **R@10** | **R@1**  | **R@5**      | **R@10** | **R@1**  | **R@5**       | **R@10** | **0.25m/2° ** | **0.5m/5° **             | **5.0m/10º** | **0.25m/2° ** | **0.5m/5° **         | **5.0m/10º** |
|-----------------------|-----------|---------|----------|--------------|----------|----------|---------------|----------|----------|--------------|----------|----------|---------------|----------|---------------|--------------------------|--------------|---------------|----------------------|--------------|

| NetVLAD 64 Pittsburgh | N         | 32768   | 47.7     | 62.8         | 70.9     | 30.7     | 41.9          | 46.4     | 82.1     | 91.4         | 93.8     | 62.2     | 73.7          | 78.4     | 5.6           | 22.0                     | 71.0         | 10.7          | 27.8                 | 84.1         |
| NetVLAD 64 Pittsburgh | Y         | 4096    | 70.7     | 81.4         | 84.6     | 30.6     | 41.9          | 47.5     | 83.7     | 91.8         | 94.0     | 67.0     | 77.8          | 80.3     | 5.8           | 23.1                     | 73.2         | 11.6          | 30.3                 | 87.5         |
| NetVLAD 64 MSLS       | N         | 32768   | 44.6     | 61.1         | 66.4     | 28.8     | 44.0          | 50.7     | 40.4     | 64.5         | 74.2     | 11.4     | 24.1          | 31.4     | 2.0           | 9.2                      | 45.5         | 1.3           | 4.5                  | 31.9         |
| NetVLAD 64 MSLS       | Y         | 4096    | 70.1     | 80.8         | 84.9     | 45.1     | 58.8          | 63.7     | 68.6     | 84.7         | 88.9     | 34.0     | 47.6          | 57.1     | 4.2           | 18.0                     | 68.1         | 3.9           | 12.1                 | 58.4         |
| NetVLAD 16 MSLS       | N         | 8192    | 49.5     | 65.0         | 71.8     | 29.3     | 43.5          | 50.4     | 48.7     | 70.6         | 78.9     | 13.0     | 33.0          | 43.8     | 1.8           | 9.2                      | 48.4         | 1.7           | 5.5                  | 39.1         |
| NetVLAD 16 MSLS       | Y         | 4096    | 70.5     | 81.1         | 84.3     | 39.4     | 53.0          | 57.5     | 70.3     | 84.1         | 89.1     | 37.8     | 53.3          | 61.0     | 4.8           | 17.9                     | 65.3         | 4.4           | 13.7                 | 61.4         |
| Patch NetVLAD         | Y         | 4096    | 79.5     | 86.2         | 87.7     | 48.1     | 57.6          | 60.5     | 88.7     | 94.5         | 95.9     | 95.9     | 96.8          | 97.1     | **9.6**       | **35.3**                 | **90.9**     | **11.8**      | **36.2**             | **96.2**     |
| AP-GeM                |           | 2048    | 64.1     | 75.0         | 78.2     | 33.7     | 44.5          | 49.4     | 80.7     | 91.4         | 94.0     | 11.4     | 22.9          | 30.5     | 5.1           | 20.5                     | 66.1         | 4.9           | 14.7                 | 65.2         |
| NetVLAD-SARE          | Y         | 4096    | 68.1     | 77.3         | 82.4     | 34.4     | 44.3          | 48.8     | **87.8** | **94.3**     | **95.9** | **79.7** | **86.7**      | **90.5** | 7.4           | 26.5                     | 81.3         | 6.4           | 19.4                 | 75.5         |
|-----------------------|-----------|---------|----------|--------------|----------|----------|---------------|----------|----------|--------------|----------|----------|---------------|----------|---------------|--------------------------|--------------|---------------|----------------------|--------------|
| VGG-GeM-GCL           | N         | 512     | 65.9     | 77.8         | 81.4     | 41.7     | 55.7          | 60.6     | 61.6     | 80.0         | 86.0     | 34.0     | 51.1          | 61.3     | 3.7           | 15.8                     | 59.7         | 3.6           | 11.2                 | 55.8         |
| VGG-GeM-GCL           | Y         | 512     | 72.0     | 83.1         | 85.8     | 47.0     | 60.8          | 65.5     | 73.3     | 85.9         | 89.9     | 47.6     | 61.0          | 69.2     | 5.4           | 21.9                     | 69.2         | 5.7           | 17.1                 | 66.3         |
| ResNet50-GeM-GCL      | N         | 2048    | 66.2     | 78.9         | 81.9     | 43.3     | 59.1          | 65.0     | 72.3     | 87.2         | 91.3     | 44.1     | 61.0          | 66.7     | 2.9           | 14.0                     | 58.8         | 3.8           | 11.8                 | 61.6         |
| ResNet50-GeM-GCL      | Y         | 1024    | 74.6     | 84.7         | 88.1     | 52.9     | 65.7          | 71.9     | 79.9     | 90.0         | 92.8     | 58.7     | 71.1          | 76.8     | 4.7           | 20.2                     | 70.0         | 5.4           | 16.5                 | 69.9         |
| ResNet152-GeM-GCL     | N         | 2048    | 70.3     | 82.0         | 84.9     | 45.7     | 62.3          | 67.9     | 72.6     | 87.9         | 91.6     | 34.0     | 51.8          | 60.6     | 2.9           | 13.1                     | 63.5         | 3.6           | 11.3                 | 63.1         |
| ResNet152-GeM-GCL     | Y         | 2048    | 79.5     | 88.1         | 90.1     | 57.9     | 70.7          | 75.7     | 80.7     | 91.5         | 93.9     | 69.5     | 81.0          | 85.1     | 6.0           | 21.6                     | 72.5         | 5.3           | 16.1                 | 66.4         |
| ResNext-GeM-GCL       | N         | 2048    | 75.5     | 86.1         | 88.5     | 56.0     | 70.8          | 75.1     | 64.0     | 81.2         | 86.6     | 37.8     | 53.6          | 62.9     | 2.7           | 13.4                     | 65.2         | 3.5           | 10.5                 | 58.8         |
| ResNext-GeM-GCL       | Y         | 1024    | **80.9** | **90.7**     | **92.6** | **62.3** | **76.2**      | **81.1** | 79.2     | 90.4         | 93.2     | 58.1     | 74.3          | 78.1     | 4.7           | 21.0                     | 74.7         | 6.1           | 18.2                 | 74.9         |


| Backbone  | Whitening | Pooling | Dimensions | Loss |  R@1 |  R@5 | R@10 | mAP@1 | mAP@5 | mAP@10 |
|-----------|-----------|:-------:|------------|:----:|:----:|:----:|:----:|:-----:|:-----:|-----------------|
| VGG       | No        |   GeM<sup>[1](https://research.mapillary.com/img/publications/CVPR20c.pdf)</sup>  | 512        |  TL  |  28  |  35  |  49  |   -   |   -   | -               |
| VGG       | No        | NetVLAD<sup>[1](https://research.mapillary.com/img/publications/CVPR20c.pdf)</sup> | 32768      |  TL  |  30  |  40  |  44  |   -   |   -   | -               |
| VGG       | No        | NetVLAD<sup>[1](https://research.mapillary.com/img/publications/CVPR20c.pdf)</sup> | 32768      |  TL  |  48  |  58  |  64  |   -   |   -   | -               |
| VGG       | No        | PatchNetVLAD<sup>[2](https://arxiv.org/abs/2103.01486)</sup> | 4096      | TL   | 48.1 | 57.6  | 60.5 |   -   |   -   |        -        |
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

Clone the [mapillary repository](https://github.com/mapillary/mapillary_sls/) and run the following command on your machine, substituting "mydir" with the path where you downloaded the mapillary library:

```shell
export MAPILLARY_ROOT="/mydir/mapillary_sls/"
```

```shell
python3 labeling/create_json_idx.py --dataset msls --root_dir /mydir/MSLS/
```

Run the extract_predictions.py script to compute the map and query features, and the top-k prediction. For instance:
```shell
python3 extract_predictions.py --dataset MSLS --root_dir /mydir/MSLS/ --subset val --model_file models/MSLS/MSLS_resnet152_GeM_480_GCL.pth --backbone resnet152 --pool GeM --f_length 2048
```
This will produce the results on the MSLS validation set for this model. If you select --subset test, the file results/MSLS/test/MSLS_resnet152_GeM_480_GCL_predictions.txt will be generated. To evaluate the predictions you will need to submit this file to the [MSLS evaluation server](https://codalab.lisn.upsaclay.fr/competitions/865#results).

To apply PCA whitening run the apply_pca.py script with the appropiate parameters. For instance, for the example above, you have to run:
```shell
python3 apply_pca.py --dataset MSLS --root_dir /mydir/MSLS/ --subset val --name MSLS_resnet152_GeM_480_GCL 
```

To reproduce all of our experiments we include a series of evaluation scripts in the scripts folder, for the MSLS, Pittsburgh, Tokyo24/7, TokyoTM, RobotCar Seasons v2 and Extended CMU Seasons datasets. These scripts need the index files for each dataset that are available [here](https://drive.google.com/drive/folders/1DT9hTiFKQH2x8aqJoFgmMGH8iftfZ0n-?usp=sharing) and our model files, available [here](https://drive.google.com/drive/folders/1RHxrAj062ZxDp5817t1s4OXGLP_i8JFX?usp=sharing).

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
