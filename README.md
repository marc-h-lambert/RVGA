# Classification

## Object

This is the companion code for the paper "Recursive variational gaussian approximation (R-VGA)". The code is available in python here. To reproduce some results just run .... To go further on this subject, a set of tutorial is also available just bellow. This tutorial introduce progressively our algorithm through the kalman filtering approach for classification. 

### [![Part1](./imgs/capsuleIcone.png)][1] Part 1: Data generation
- Trajectory generation of a 3 ddl trajectory with aerodynamic forces in a geocentric Earth frame
- Introduction to the abstract space toolbox to manage frame and change of coordinates
- A flat Earth variant of this tutorial is available here: [Tracking_Vehicle_Reentry_FlatEarth](Tracking_Vehicle_Reentry_FlatEarth.ipynb)
        
### [![Part2](./imgs/radarIcone.png)][2] Part 2: Linear regression 
- Linearization of observation from spherical coordinates to cartesian coordinates
- Simulation of the measurement noise of a planar radar sensor 
        
### [![Part3](./imgs/ballisticIcone.png)][3] Part 3: Logistic Regression
- Overview on the linear Kalman filter
- Extended Kalman filter to track a vehicle in the case where aerodynamic forces are null
- Tracking with passive sensor and multi sensor fusion for stereovision
        
### ![Part4](./imgs/filterIcone.png) Part 4: the R-VGA approach
- Set up of a Kalman filter to track a vehicle in the case where aerodynamic forces are not null

## Ressources

Tutorial are written in jupyter notebook. You can read them from Git Hub but if you want to run the code you need to install locally jupyter notebook https://jupyter.org/ and download the tutorial files (.ipynb). The code of the Tutorial is also  available in the Kalman4ClassifLibrary. To test this library run [Test_AeroSpaceLibrary.py](Test_AeroSpaceLibrary.py).

[1]: Tracking_Vehicle_Reentry_Part1.ipynb
[2]: Tracking_Vehicle_Reentry_Part2.ipynb
[3]: Tracking_Vehicle_Reentry_Part3.ipynb
