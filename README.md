# Failing CI Builds Prediction

Continuous Integration is a set of modern Software Development practices widely adopted in commercial and open source environments that aim at supporting developers in integrating code changes constantly and quickly through an automated building and testing process.
This enables them to discover newly induced errors before they reach the end-user and to fix them quicker as changes integrated in small increments are easier to debug.

However, this build process is typically time and resource-consuming, specially for large projects with many dependencies where the build process can take hours or even days. This may cause disruptions in the development process and delays in the product release dates.

One of the main causes of these delays is that some commits needlessly start the Continuous Integration process, therefore ignoring them will significantly lower the cost of Continuous Integration without sacrificing its benefits.

Thus, withing the framework of this project, we aim to reduce the number of builds executed by skipping those deemed unnecessary.

This repository contains:
- Benchmark of different Machine Learning models for CI skip commits detection on the TravisTorrent dataset.
- Using a Genetic Algorithm to detect the optimal CI-skip decision rule using Binary Tree representation of rules. implementation of [1]
- Hyper Parameter Optimization of ML/DL models using Genetic Algorithms. implementation of [2]
- A novel Deep Reinforcement Learning based approach to build an optimal decision tree that account for data imbalance to classify commits. inspired by [3]

[1] Detecting continuous integration skip commits using multi-objective evolutionary search, Islem Saidani, Ali Ouni, and Mohamed Wiem Mkaouer
[2] Improving the prediction of continuous integration build failures using deep learning, Islem Saidani, Ali Ouni, and Mohamed Wiem Mkaouer
[3] Building decision tree for imbalanced classification via deep reinforcement learning, Guixuan Wen and Kaigui Wu
