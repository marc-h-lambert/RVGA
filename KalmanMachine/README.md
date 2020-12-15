# The Kalman machine library

## Object

A library to assess Bayesian sequential algorithms based on the Kalman filtering framework.

The arborescence is the following:

- KDataGenerator: generate synthetic dataset for linear and logistic regression
- Kalman4LogisticReg: the sequential Bayesian optimizer for linear and logistic regression 
  --> include three versions for logistic regression: EKF, QKF and RVGA (in progress...)
- KEvaluation: the evaluation metrics for linear and logistic regression
