# object-based-location-classification

## Introduction

This repository contains training and evaluation code to train ANN and LSTM networks for location classification based on detected objects on a video frame.
We use Yolo for object detections but anything can be used to produce the classification input as long as the format in the train files is not modified.

The code is tested and working on PyTorch 0.4.0 on Windows 10.

## Installation

Clone the repository to your computer
``` 
git clone https://github.com/georkap/object-based-location-classification 
```
We include the object detections for the 20 object classes of the adl dataset for detection threshold 0.3 (case d2d 0.3) for all the videos of the ADL dataset.

## Dependencies
* pytorch 0.4
* numpy
* sklearn

## Licence
Our code and data are available under the MIT licence. If you use them for scientific research please consider citing our paper:
```
@inproceedings{kapidis_where_2018-1,
  title = {Where {{Am I}}? {{Comparing CNN}} and {{LSTM}} for {{Location Classification}} in {{Egocentric Videos}}},
  doi = {10.1109/PERCOMW.2018.8480258},
  booktitle = {2018 {{IEEE International Conference}} on {{Pervasive Computing}} and {{Communications Workshops}} ({{PerCom Workshops}})},
  author = {Kapidis, G. and Poppe, R. W. and van Dam, E. A. and Veltkamp, R. C. and Noldus, L. P. J. J.},
  month = mar,
  year = {2018},
  pages = {878-883}
}
```

## Contact
g.kapidis{at}uu.nl
