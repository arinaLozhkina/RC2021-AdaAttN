# AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer

## Project Description 
In this project we reproduce (as a part of [Reproducibility Challenge 2021](https://paperswithcode.com/rc2021))  implementation of style transfer based on the paper [AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer, S. Liu et al. 2021](https://arxiv.org/abs/2108.03647). The goal of this project is to reimplement the style transfer model from scratch. 
![Alt text](./fullModel.png?raw=true "Overall Model")


## Code Description 
To reproduce a paper we use PyTorch framework, all required packages are mentioned in preretirements.txt. 
The repository contains files: 
| Path to file | Description |
| --- | --- |
| /src/simple_dataset.py | contains dataset implementation with fixed length | 
| /src/train.py | contains class TrainModel to run training of a model |
| /src/models.py | contains classes of encoder, AdaAttN, decoder and overall model | 
| /src/loss.py | contains custom loss function used in the paper |
| prerquieremtns.txt | contains all packages needed to run training |

## How to Install and Run the Project 
In order to run training: 
1. Clone the repository: 
```
gh repo clone arinaLozhkina/RC2021-AdaAttN-
```
2. Add images for style and content images:  
```
mkdir data 
cd ./data 
wget "COCO dataset" -O temp.zip
unzip temp.zip -d ./train2014
rm temp.zip
wget "WikiArt dataset" -O temp.zip
unzip temp.zip -d ./wikiart
rm temp.zip
```
3. Install packages
```
pip3 install -r ../requirements.txt 
```
4. Run train.py 
```
python3 train.py 
```

## Report 
The paper’s reproducibility report is report.pdf. 
