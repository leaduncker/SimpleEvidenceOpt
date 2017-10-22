# SimpleEvidenceOpt

This repository contains code to perform simple Emprirical Bayesian Evidence Optimization for Spatio-Temporal receptive fields. The code flexibly combines different covariance functions for the spatial and temporal component of the receptive field.

## Available Covariance Functions
* Automatic Smoothness Determination [1]
* Automatic Locality Determination [2]
* Ridge Regression 
* Temporal Recency Determination [3]

## Dependencies
* Matlab 2016b or newer


## References
[1] M Sahani, and J F Linden. Evidence optimization techniques for estimating stimulus-response functions. Advances in neural information processing systems. 2003.

[2] M Park and J W Pillow, Receptive field inference with localized priors. PLoS computational biology, 7(10), 2011, p.e1002219.

[3] L Duncker, S Ravi, G D Field, J W Pillow. Scalable variational inference for low-rank receptive fields with non-stationary smoothness.
Computational and Systems Neuroscience (CoSyNe), 2017.
