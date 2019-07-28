# Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN, PyTorch)
This repository is a work in progress and is updated regularly. Documentation and usage may be unclear right now.

This repository is an implementation of [this paper](https://arxiv.org/abs/1612.07828) in PyTorch. Many of the repositories I found that were also written in PyTorch either were (1) buggy or (2) incomplete. This repository modifies other repos and makes this implementation more complete and hopefully easier to use. 

## Motivation
This repository was completed over the course of my summer internship with Ford. [This paper](https://arxiv.org/abs/1612.07828) presents a method to add realism to unreal or synthetic data, allowing us to train on a labeled set of synthetic data that matches the distribution of realistic data 'better,' resulting in better testing accuracy. Fix later.


## Overview
What is this project. What does it do? How does it work? Fix image sizes. Give brief explanations to the project. Fix later.

![the_refiner](https://github.ford.com/DMERRIC5/Learning-from-Simulated-and-Unsupervised-Images-through-Adversarial-Training-SimGAN-PyTorch-/blob/master/images/refiner.png)

![network_diagram](https://github.ford.com/DMERRIC5/Learning-from-Simulated-and-Unsupervised-Images-through-Adversarial-Training-SimGAN-PyTorch-/blob/master/images/network_architecture.png)

## Build status
In development! Documentation and usage information is currently being uploaded. Results will be added in a few days.

## Results
Columns 1, 3, 5 and 7 are the input syntheric images (Unity Eyes). 
Columns 2, 4, 6 and 8 are the refined images (Unity Eyes + Realism)

Learning Rate: 0.001
K_R: 2
Delta (penalty on reconstuction loss): 0.75
Batch Size: 512
Buffer Size: 128000
Num Steps: 100000

![Results](https://github.com/dmerrick520/Learning-from-Simulated-and-Unsupervised-Images-through-Adversarial-Training-SimGAN-PyTorch/blob/master/images/001_2_P75_512_128000_100000.jpg)

## Installation
Provide step by step series of examples and explanations about how to get a development env running.
... Going to include docker files ... Fix later.

## Usage
If people like your project they’ll want to learn how they can use it. To do so include step by step guide to use your project.
Include links to the sub READMEs in the gaze_estimator and simgan folders. Fix later.

[This is the link to the gaze_estimator README with instructions on usage](/gaze_estimator/README.md)

[This is the link to the simgan README with instructions on usage](/simgan/README.md)

## Contribute
Let people know how they can contribute into your project. A [contributing guideline](https://github.com/zulip/zulip-electron/blob/master/CONTRIBUTING.md) will be a big plus. Fix later.

## Credits
Give proper credits. This could be a link to any repo which inspired you to build this project, any blogposts or links to people who contrbuted in this project. Fix later.

[This is the paper that this repo is based off of](https://arxiv.org/abs/1612.07828)
This repo is largely based on [this guys repo](https://github.com/automan000/SimGAN_PyTorch) but I fixed some of the bugs and reorganized the code so that it is hopefully easier to use.

#### Anything else that seems useful
