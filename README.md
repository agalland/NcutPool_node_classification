# Graph Neural Network Pooling By Edge Cut
# Version for node classification

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper, run this command:

```train
python main.py```

The setting for the split train, test, validation is the same as the one described in the experiments. The hyper-parameters selected are the one that produce the best classification accuracy.


## Evaluation

To evaluate my model on Cora, run:

```eval
python evaluate.py
```

A random fold was selected and train on the molecules of the train index. The accuracies are displayed for the train, test and validation sets.
Train, test and validation indices are stores in "weightsEval/" as well. The parameters are the ones of the file evaluate.py

## Pre-trained Models

You can download pretrained models here:

- weights are included in the zip file in the weightsEval folder


## Results

Our model achieves the following performance on graph classification datasets:

### [Graph Classification on PROTEINS, DD and COLLAB]

| Model name  |     Cora    |    Citeseer   |    Pubmed   |
| ------------|-------------| ------------- | ----------- |
|   EdgeCut   |    82.3%    |     70.9%     |     79.1%   |



