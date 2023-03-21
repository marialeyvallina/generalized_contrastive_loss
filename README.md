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
  title={Data-efficient Large Scale Place Recognition with Graded Similarity Supervision}, 
  author={María Leyva-Vallina and Nicola Strisciuglio and Nicolai Petkov},
  journal={CVPR},
  year={2023}
}
```
## Network activation
![](https://github.com/marialeyvallina/generalized_contrastive_loss/blob/main/attention.png)

## Contact details
If you have any doubts please contact us at:
1. María Leyva-Vallina: m.leyva.vallina at rug dot nl
2. Nicola Strisciuglio: n.strisciuglio at utwente dot nl
## How to use this library
### Download the data
1. MSLS: The dataset is available on request [here](https://www.mapillary.com/dataset/places "MSLS"). For the new GT annotations, please request them [here](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/W4LIGP&faces-redirect=true).
2. Pittsburgh: The whole dataset is available on request [here](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/ "Pittsburgh") and the train val splits for Pitts30k are available [here](https://www.di.ens.fr/willow/research/netvlad/ "Pitts30k"). 
3. TokyoTM: The dataset is available on request [here](https://www.di.ens.fr/willow/research/netvlad/ "TokyoTM"). 
4. Tokyo 24/7: The dataset is available on request [here](http://www.ok.ctrl.titech.ac.jp/~torii/project/247/ "Tokyo 24/7"). 
5. TB-Places: The dataset is available [here](https://github.com/marialeyvallina/TB_Places "TB-Places"). For the new GT annotations, please request them [here](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/W4LIGP&faces-redirect=true).

### Download the models
All our models can be downloaded from request them [here](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/W4LIGP&faces-redirect=true).
### Our results
#### MSLS

|                   |           |         |          | **MSLS-val** |          |          | **MSLS-test** |          |          | **Pitts30k** |          |          | **Tokyo24/7** |          |              | **RobotSeasons v2- all** |              |              | **Extended CMU-all** |              |
|-------------------|-----------|---------|----------|--------------|----------|----------|---------------|----------|----------|--------------|----------|----------|---------------|----------|--------------|--------------------------|--------------|--------------|----------------------|--------------|
| **Method**        | **PCA_w** | **Dim** | **R@1**  | **R@5**      | **R@10** | **R@1**  | **R@5**       | **R@10** | **R@1**  | **R@5**      | **R@10** | **R@1**  | **R@5**       | **R@10** | **0.25m/2°** | **0.5m/5º**              | **5.0m/10º** | **0.25m/2°** | **0.5m/5º**          | **5.0m/10º** |
| NetVLAD-GCL       | N         | 32768   | 62.7     | 75.0         | 79.1     | 41.0     | 55.3          | 61.7     | 52.5     | 74.1         | 81.7     | 20.3     | 45.4          | 49.5     | 3.3          | 14.1                     | 58.2         | 3.0          | 9.7                  | 52.3         |
| NetVLAD-GCL       | Y         | 4096    | 63.2     | 74.9         | 78.1     | 41.5     | 56.2          | 61.3     | 53.5     | 75.2         | 82.9     | 28.3     | 41.9          | 54.9     | 3.4          | 14.2                     | 58.8         | 3.1          | 9.7                  | 52.4         |
| VGG-GeM-GCL       | N         | 512     | 65.9     | 77.8         | 81.4     | 41.7     | 55.7          | 60.6     | 61.6     | 80.0         | 86.0     | 34.0     | 51.1          | 61.3     | 3.7          | 15.8                     | 59.7         | 3.6          | 11.2                 | 55.8         |
| VGG-GeM-GCL       | Y         | 512     | 72.0     | 83.1         | 85.8     | 47.0     | 60.8          | 65.5     | 73.3     | 85.9         | 89.9     | 47.6     | 61.0          | 69.2     | 5.4          | 21.9                     | 69.2         | 5.7          | 17.1                 | 66.3         |
| ResNet50-GeM-GCL  | N         | 2048    | 66.2     | 78.9         | 81.9     | 43.3     | 59.1          | 65.0     | 72.3     | 87.2         | 91.3     | 44.1     | 61.0          | 66.7     | 2.9          | 14.0                     | 58.8         | 3.8          | 11.8                 | 61.6         |
| ResNet50-GeM-GCL  | Y         | 1024    | 74.6     | 84.7         | 88.1     | 52.9     | 65.7          | 71.9     | 79.9     | 90.0         | 92.8     | 58.7     | 71.1          | 76.8     | 4.7          | 20.2                     | 70.0         | 5.4          | 16.5                 | 69.9         |
| ResNet152-GeM-GCL | N         | 2048    | 70.3     | 82.0         | 84.9     | 45.7     | 62.3          | 67.9     | 72.6     | 87.9         | 91.6     | 34.0     | 51.8          | 60.6     | 2.9          | 13.1                     | 63.5         | 3.6          | 11.3                 | 63.1         |
| ResNet152-GeM-GCL | Y         | 2048    | 79.5     | 88.1         | 90.1     | 57.9     | 70.7          | 75.7     | **80.7** | **91.5**     | **93.9** | **69.5** | **81.0**      | **85.1** | **6.0**      | **21.6**                 | 72.5         | 5.3          | 16.1                 | 66.4         |
| ResNeXt-GeM-GCL   | N         | 2048    | 75.5     | 86.1         | 88.5     | 56.0     | 70.8          | 75.1     | 64.0     | 81.2         | 86.6     | 37.8     | 53.6          | 62.9     | 2.7          | 13.4                     | 65.2         | 3.5          | 10.5                 | 58.8         |
| ResNeXt-GeM-GCL   | Y         | 1024    | **80.9** | **90.7**     | **92.6** | **62.3** | **76.2**      | **81.1** | 79.2     | 90.4         | 93.2     | 58.1     | 74.3          | 78.1     | 4.7          | 21.0                     | **74.7**     | **6.1**      | **18.2**             | **74.9**     |

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


### Train your own models
If you want to train a model for MSLS using the GCL function you must execute train.py with the appropiate parameters. For example:
```shell
python3 train.py --root_dir mydir/MSLS/ --cities val --backbone vgg16 --use_gpu --pool GeM --last_layer 2 
```
Make sure that your root dir contains the graded GT files.
