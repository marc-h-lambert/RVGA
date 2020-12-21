# The Kalman machine library

## python files

A library to assess Bayesian sequential algorithms based on the Kalman filtering framework.

The arborescence is the following:

- [KDataGenerator][1]: generate synthetic dataset for linear and logistic regression
- [BayesianLogisticReg][2]: the Bayesian framework for logistic regression: include Laplace approximation 
- [Kalman4LogisticReg][3]: the online Bayesian algorithms for logistic regression: include three versions 
for logistic regression: EKF, QKF and RVGA 
- [KEvalPosterior][4]: the evaluation metrics to assess posterior estimation for logistic regression

[1]: ./KDataGenerator.py
[2]: ./BayesianLogisticReg
[3]: ./Kalman4LogisticReg.py
[4]: ./KEvalPosterior.py
