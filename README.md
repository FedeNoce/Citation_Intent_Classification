# Citation_Intent_Classification


## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
  * [Installation](#installation)
* [Usage](#usage)
* [Authors](#authors)



## About The Project


Citation Intent Classification in scientific papers using the Scicite dataset an Pytorch

For more information, read the [report](report.pdf) located in the repo
root.

### Built With

* [Python](https://www.python.org/)
* [Pytorch](https://pytorch.org/)
* [GloVe](https://nlp.stanford.edu/projects/glove/)
* [ELMo](https://allenai.org/allennlp/software/elmo)
* [Scicite](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz)





### Installation
To get a local copy up and running follow these simple steps.

1. Clone the repo
```sh
git clone https://github.com/FedeNoce/Citation_Intent_Classification.git
```
2. Download [GloVe](https://nlp.stanford.edu/projects/glove/) and [ELMo](https://allenai.org/allennlp/software/elmo) word representation from the link above
3. Download [Scicite](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz) dataset


## Usage

1. Set hyperparameters and paths in ```classification.py``` and in ```utils.py```

2. To train the model run ```classification.py```

## Authors

* [**Federico Nocentini**](https://github.com/FedeNoce)
* [**Corso Vignoli**](https://github.com/CVignoli)


