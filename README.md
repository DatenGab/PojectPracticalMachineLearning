# PojectPracticalMachineLearning


Executive Summary

This is the course project of the Practical Machine Learning Curse. The report develops how the goals of this projects are acomplished: 1) The data is cleaned to avoid using NA variables. 2) The 19622 experiments for training are divided by 70/30 for create the model and for test the results and for measure the accuracy. 3) A first model using classification tree is created, but the accuracy is not enought. 4) A final model is created using random forest which computes a 99% of accuracy, which is requiered to obtain a 95% of confidence for predincting 20 cases. In order to improve the performance, the model is training using a k-fold=5 and processing in parallel. 5) As the accuracy of the used model is of 99%, we predict the 20 cases with a 95% of confidence.
Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
Loading Libraries and reading data

First, the required library are loaded and the input data is read.

library(caret)

## Loading required package: lattice

## Loading required package: ggplot2

library(rpart)
library(rattle)

## Rattle: A free graphical interface for data science with R.
## Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.

library(parallel)
library(doParallel)

## Loading required package: foreach

## Loading required package: iterators

pml_training = read.csv("~/Desktop/pml-training.csv",  na.strings = c("NA", "#DIV/0!", ""), header = TRUE)
pml_testing = read.csv("~/Desktop/pml-testing.csv",na.strings = c("NA", "#DIV/0!", ""), header = TRUE)
dim(pml_training)

## [1] 19622   160

dim(pml_testing)

## [1]  20 160

Cleaning Data

There are several variables (columns) with NA value. These colums are removed using the function is.na to test if the sum of column is or not NA before removing

training1<- pml_training[,colSums(is.na(pml_training)) == 0]
testing1<- pml_testing[,colSums(is.na(pml_testing)) == 0]

The first seven columns are removed before they give information about the people who did the test, and timestamps, which are not related with the classification we are trying to predict.

training<- training1[,-c(1:7)]
testing<- testing1[,-c(1:7)]
dim(training)

## [1] 19622    53

dim(testing)

## [1] 20 53

#how many sambles we have for each classe
table(training$classe)

## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607

There are 19622 experiments with 53 variables for training and validation of our models, and 20 rows for testing
Data Partition

The training set is used for training and for validation, in 70/30 proportion.

inTrain = createDataPartition(training$classe, p = 0.70)[[1]]
training_part = training[ inTrain,]
valid_part = training[-inTrain,]

Predictive Model using classification trees

A classification tree model is created using 13737 experiments of the training set. The tree is plotted.

model_CT <- train(classe~., data=training_part, method="rpart")
fancyRpartPlot(model_CT$finalModel)

We predict values using the valid set and we calculate the confussion matrix with the accurary results.

predict_validation<- predict(model_CT, newdata = valid_part)
cm_ct<-confusionMatrix(predict_validation,valid_part$classe)
cm_ct$cm_ct$overall['Accuracy']

## NULL

The accuracy result is low, of 49% with a 95% CI of(48%-50%).
Predictive Model using Random Forest

We create a new model using random forest. As the training would be very slow, I follow the instructions of the next link https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md. A cluster is created and the resampling method is changing for using k-fold cross-validation with number=5.

#use k_fold=5  in cross_validation to improve the performance
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
trainControl_function <-trainControl(method = "cv",number = 5, allowParallel = TRUE) 
model_rf <- train(classe~., data=training_part, method="rf",trControl = trainControl_function)
print(model_rf$finalmodel)

## NULL

##stop of paralling computing.
stopCluster(cluster)  
registerDoSEQ()

We predict values of valid set and calculate the confussion matrix with the accurary results.

predict_validation_rf<- predict(model_rf, newdata = valid_part)
cm_rf<-confusionMatrix(predict_validation_rf,valid_part$classe)
cm_rf$overall['Accuracy']

## Accuracy 
## 0.993373

The accuracy result is 99%, enough to get the prediction of the 20 values. As you can see in the next entry, this is the accuracy required to obtain a 95% of confidence in the prediction of 20 values. https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-requiredModelAccuracy.md

This is the plot of the model error rate by number of trees and 20 most important variables (out of 52)

plot(model_rf$finalModel,main="Random forest model error rate by number of trees")

varImp(model_rf)

## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                   Overall
## roll_belt          100.00
## yaw_belt            79.88
## magnet_dumbbell_z   69.14
## pitch_belt          61.42
## pitch_forearm       61.07
## magnet_dumbbell_y   61.03
## roll_forearm        49.86
## magnet_dumbbell_x   49.48
## accel_dumbbell_y    43.64
## accel_belt_z        42.51
## magnet_belt_z       41.49
## roll_dumbbell       40.02
## magnet_belt_y       39.69
## accel_dumbbell_z    36.27
## roll_arm            36.00
## accel_forearm_x     34.19
## gyros_belt_z        30.62
## yaw_dumbbell        29.47
## accel_dumbbell_x    28.36
## gyros_dumbbell_y    27.16

Predicting using the test set

The random forest model is now used to predict the manner in which the people will do the exercise. The final results are saved in a file.

predict_test<- predict(model_rf, testing)
predict_test

##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E

write.csv(predict_test,"~/Desktop/result.csv"
