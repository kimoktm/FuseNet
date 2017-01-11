# FuseNet
Code for reproducing main results in the paper [FuseNet: FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture](https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf) by C. Hazirbas and L. Ma and C. Domokos and D. Cremers.

<img src="Images/framework.png" width="800px" height="386px"/>

### Dependencies
- python 2.7
- [TensorFlow 0.12 GPU version](https://www.tensorflow.org/get_started/os_setup)
- In addition, please `pip install -r requirements.txt` to install the following packages:
    - `Pillow`
    - `h5py`
    - `joblib`
    - `scipy`

**Data**
1. Download NYUv2 RGB-D dataset [here](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). This dataset contains 1449 paired RGB-D images. To map the original labels into 40 classes, we use the mapping by Gupta et al from this [paper](https://people.eecs.berkeley.edu/~sgupta/pdf/GuptaArbelaezMalikCVPR13.pdf). For your convenience, we put the data split and label mapping in the subfolders `Raw/`.
```
nyuv2_40_annots_mapping.mat      maps to 40 annotations, 0 is invalid
nyuv2_10_class_mapping.mat       maps to 10 classes (scenes), 0 is invalid
annotations_names.txt            mapped annotations names (indexed according to mapping)
class_names.txt                  mapped classes (scenes) names (indexed according to mapping)
splits.mat                       standard train-test split
```
2. Preprocess raw dataset and save results to `Datasets/NYU` using
  `python data/nyu_dataset_extract.py -i nyu_dataset.mat -s splits.mat -l mapping.mat -o Datasets/NYU`
3. Convert extracted dataset to tfrecords `Datasets/NYU/tfrecords` using
  `python data/dataset_to_tfrecords.py --train_dir NYU/train --test_dir NYU/test --output_dir NYU/tfrecords --classes_file NYU/class_names.txt`


### Training
- To train Fusenet run `fusenet_train` passing tfrecords dir
  `python fusenet_train.py`


### To-Do
- [ ] Adjust nyu_extractor to integrate validation split as well
- [ ] Add Weight decay to training
- [ ] Add decay rate to learning (multiply by 0.9 in every 50,000 iterations)
- [ ] Use Weights initalization from VGG-16
- [ ] Add Accuray measurement for evalution
- [ ] Add `fusenet/fusenet_eval.py` to evalute and visualize prediction


### Citing FuseNet
Caner Hazirbas, Lingni Ma, Csaba Domokos and Daniel Cremers, _"FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-based CNN Architecture"_, in proceedings of the 13th Asian Conference on Computer Vision, 2016. ([pdf](https://vision.in.tum.de/_media/spezial/bib/hazirbasma2016fusenet.pdf))

    @inproceedings{fusenet2016accv,
     author    = "C. Hazirbas and L. Ma and C. Domokos and D. Cremers",
     title     = "FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture",
     booktitle = "Asian Conference on Computer Vision",
     year      = "2016",
     month     = "November",
    }