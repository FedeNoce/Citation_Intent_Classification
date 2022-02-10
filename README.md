# Citation_Intent_Classification


## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)
* [Authors](#authors)
* [Acknowledgments](#acknowledgments)


## About The Project

![Demo](video/demo.gif)


Citation Intent Classification in scientific papers using the Scicite dataset an Pytorch

For more information, read the [report](report.pdf) located in the repo
root.

### Built With

* [Python](https://www.python.org/)
* [Pytorch](https://pytorch.org/)
* [GloVe](https://nlp.stanford.edu/projects/glove/)
* [ELMo](https://allenai.org/allennlp/software/elmo)



## Getting Started

To get a local copy up and running follow these simple steps.

### Installation
 
1. Clone the repo
```sh
git clone https://github.com/FedeNoce/Citation_Intent_Classification.git
```
2. Download GloVe and ELMo word representation from the link above




## Usage

1. If you want to calibrate your camera print a [chessboard](https://miro.medium.com/max/700/1*C5b8iGTkcgmLZccfeklcmw.png), take at least 20 photos from different angles and distances and copy them in a folder. Run ```camera_calibration.py``` modifying the paths and the chessboard parameters as needed

2. If needed change the ```frame_to_skip``` parameter in ```VideoGet.py``` to skip an arbitrary number of frames to reduce the latency

3. Run ```social_distance_detector.py``` and choose the input video stream (optionally adding the calibration matrices path) between the available ones:
   - Computer webcam
   - IP webcam
   - Local video (some examples are in ```/video```)
   - Link to stream (some examples are in ```webcam_stream.txt```)

## Authors

* [**Lorenzo Agnolucci**](https://github.com/LorenzoAgnolucci)
* [**Federico Nocentini**](https://github.com/FedeNoce)


## Acknowledgments
Image and Video Analysis © Course held by Professor [Pietro Pala](https://scholar.google.it/citations?user=l9j1pZEAAAAJ&hl=en) - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html)

