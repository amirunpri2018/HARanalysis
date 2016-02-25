---
title: "Human Activity Recognition"
author: "Burak H."
date: "February 24, 2016"
output: html_document
---

## Background
In the following, we analyze the data from the accelerometers from the belt, forearm, arm, and dumbell of 6 participants. The aim is to correctly predict the way each activity is performed, divided in 5 classes (sitting-down, standing-up, standing, walking, and sitting). 

## Cleaning the data

We first read the training and test data

```r
tr0 <- read.csv("pml-training.csv",na.strings=c("NA",""))
ts0 <- read.csv("pml-testing.csv",na.strings=c("NA",""))
```
Then, we remove all the columns that are completely NA, using the following

```r
tr1 <- tr0[, colSums(is.na(tr0)) == 0]
ts1 <- ts0[, colSums(is.na(ts0)) == 0]
```
We also choose to remove some of the variables, such as timestamps, user_names etc.

```r
tr1 <- select(tr1, -c(X ,user_name, raw_timestamp_part_1,
                       raw_timestamp_part_2, cvtd_timestamp, 
                       new_window, num_window) )
ts1 <- select(ts1, -c(X ,user_name, raw_timestamp_part_1,
                      raw_timestamp_part_2, cvtd_timestamp, 
                      new_window, num_window) )
```

## Simple Decision Tree
Let us first divide the tr1 set into training and test sets for developing our model:

```r
suppressMessages(suppressWarnings(library(caret)))
suppressMessages(suppressWarnings(library(tree)))
```

```r
set.seed(101)
inTrain <- createDataPartition(y = tr1$classe, p = 0.7, list = FALSE)
training <- tr1[inTrain, ]; testing <- tr1[-inTrain, ]
```
Now, let us train a simple tree:

```r
tree.har=tree(classe~.,data=training)
```
The resulting tree is shown below:

![Decision Tree](figure/unnamed-chunk-8-1.png) 

This tree is quite bushy, and therefore prone to overfitting. Let us find the predictions based on this tree:


```r
tree.pred <- predict(tree.har, newdata = testing, type = "class")
cm.tree <- confusionMatrix(tree.pred, testing$classe); round(cm.tree$overall,2)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##           0.68           0.60           0.67           0.70           0.28 
## AccuracyPValue  McnemarPValue 
##           0.00           0.00
```
The accuracy of the this decision tree is about 0.68. Let us try to use cross validation to prune the tree and reduce possible overfitting. We choose 10-fold cross validation to train a new tree:


```r
set.seed(201)
ctrl <- trainControl(method = "cv", number = 10)
tree.har <- train(classe ~., data = training, method = "rpart", trControl = ctrl)
```

Surprisingly, this new (pruned) tree performs worse than the bushy one:

```r
tree.pred <- predict(tree.har$finalModel, newdata = testing, type = "class")
cm.tree <- confusionMatrix(tree.pred, testing$classe); round(cm.tree$overall,2)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##           0.50           0.34           0.48           0.51           0.28 
## AccuracyPValue  McnemarPValue 
##           0.00            NaN
```

Therefore, we choose to use Random Forests in order to train a more accurate model.

## Random Forest Analysis
Let us train a random forest model, with 5-fold cross validation and 200 trees. 


```r
ctrl <- trainControl(allowParallel=T, method="cv", number=5)
rf.har <- train(classe ~., data = training, method = "rf", 
                           ntree=200, trControl = ctrl)
rf.har$results
```

```
##   mtry  Accuracy     Kappa  AccuracySD     KappaSD
## 1    2 0.9892261 0.9863698 0.001424866 0.001802034
## 2   27 0.9898811 0.9871995 0.001917181 0.002425168
## 3   52 0.9804902 0.9753169 0.003202961 0.004049993
```
This chosen random forest model has mtry=27 (number of randomly chosen variables at each branch) and is very accurate (about %99).
Using this model, we predict on the testing set:

```r
rf.predict <- predict(rf.har, newdata = testing)
cm.rf <- confusionMatrix(rf.predict, testing$classe); round(cm.rf$overall,2)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##           0.99           0.99           0.99           0.99           0.28 
## AccuracyPValue  McnemarPValue 
##           0.00            NaN
```
Finally, we predict on the original testing set (where classe is unknown)

```r
rf.predict.ts <- predict(rf.har, newdata = ts1)
rf.predict.ts
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
