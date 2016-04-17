# Reproducible Research - Course Project 1

## Introduction

The dataset for this project was created by the source data set in the files folder in this repository. The data was provided from the Coursera course website. 

## About the source data

The data recorded with four 9 degrees of freedom Razor inertial measurement units (IMU), which provide three-axes acceleration, gyroscope and magnetometer data at a joint sampling rate of 45 Hz. Each IMU also featured a Bluetooth module to stream the recorded data to a notebook running the Context Recognition Network Toolbox. The sensors in the users’ glove, armband, lumbar belt and
dumbbell. 

Participants performed one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 

* exactly according to the specification (Class A) 
* throwing the elbows to the front (Class B) 
* lifting the dumbbell only halfway (Class C) 
* lowering the dumbbell only halfway (Class D)
* throwing the hips to the front (Class E) 

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. 


The training dataset is stored in a comma-separated-value (CSV) file and there are a total of 19,622 observations in this dataset.
The testing dataset is stored in a comma-separated-value (CSV) file and there are a total of 20 observations in this dataset.


## Feature extration in the original data set

Features were extracted used a sliding window approach with different lengths from 0.5 second to 2.5 seconds, with 0.5 second overlap. In each step of the sliding window approach features were calculated on the Euler angles (roll, pitch
and yaw), as well as the raw accelerometer, gyroscope and magnetometer readings. For the Euler angles of each of the
four sensors eight features were extracted: mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness.

## Transformations on the dataset

* Variables with more than 95% of NA were removed from the data set
* Zero and Near Zero variables were removed from the model
* The first 6 variables were removed 


## Acknowledgements

This assignment uses data from the <a href="https://www.coursera.org/">Coursera website</a> - Practical Machine Learning and from <a href="http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises">Human Activity Recognition website</a>.







