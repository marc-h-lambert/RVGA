# The Kalman machine library

## Object

A library to assess Bayesian sequential algorithms based on the Kalman filtering framework.

The arborescence is the following:

- [KDataGenerator][1]: generate synthetic dataset for linear and logistic regression
- [Kalman4LogisticReg][2]: the sequential Bayesian optimizer for linear and logistic regression 
  --> include three versions for logistic regression: EKF, QKF and RVGA (in progress...)
- [KEvaluation][3]: the evaluation metrics for linear and logistic regression
- [KUtils][4]: the evaluation metrics for linear and logistic regression

[1]: ./KDataGenerator.py
[2]: ./Kalman4LogisticReg.py
[3]: ./KEvaluation.py
[4]: ./KUtils.py
