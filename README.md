# Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN, PyTorch)
This repository is finished but documentation is being updated. Documentation and usage may be unclear right now, any questions or concerns are welcome!.

This repository is an implementation of [this paper](https://arxiv.org/abs/1612.07828) in PyTorch. Many of the repositories I found that were also written in PyTorch either were (1) buggy or (2) incomplete. This repository modifies other repos and makes this implementation more complete and hopefully easier to use. 

## Motivation
[This paper](https://arxiv.org/abs/1612.07828) presents a method to add realism to unreal or synthetic data, allowing us to train on a labeled set of synthetic data that matches the distribution of realistic data 'better,' resulting in better testing accuracy. Finish later.

## Build status
Code base is finished! Documentation and usage information is being uploaded. More detailed results will be added eventually.

## Results
Columns 1, 3, 5 and 7 are the input syntheric images (Unity Eyes). <br/>
Columns 2, 4, 6 and 8 are the refined images (Unity Eyes + Realism)

Learning Rate: 0.001 <br/>
K_R: 2 <br/>
Delta (penalty on reconstuction loss): 0.75 <br/>
Batch Size: 512 <br/>
Buffer Size: 128000 <br/>
Num Steps: 100000 <br/>

![Results](https://github.com/dmerrick520/Learning-from-Simulated-and-Unsupervised-Images-through-Adversarial-Training-SimGAN-PyTorch/blob/master/images/001_2_P75_512_128000_100000.jpg)

## Installation
Provide step by step series of examples and explanations about how to get a development env running.
... Going to include docker files ... Needs updating.

## Usage
Needs updating.

[This is the link to the gaze_estimator README with instructions on usage](/gaze_estimator/README.md)

[This is the link to the simgan README with instructions on usage](/simgan/README.md)


## Credits

This project is based on [this paper](https://arxiv.org/abs/1612.07828).
This repo is largely built from [automan000's repository](https://github.com/automan000/SimGAN_PyTorch) but I fixed some of the bugs, reorganized the code, and added code to train & test a gaze estimator.

#### Anything else that seems useful
