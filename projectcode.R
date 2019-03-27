
# machine learning project

library(caret)
library(randomForest)
library(gbm)
set.seed(99281)

# Enable parallel processing
library(doParallel)
cl <- makePSOCKcluster(3, outfile = 'log.txt') #leave one core open
registerDoParallel(cl)


## Download training and test data
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = './machinelearningproject/training.csv')
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = './machinelearningproject/test.csv')

training = read.csv("./machinelearningproject/training.csv")
test = read.csv("./machinelearningproject/test.csv")

## Split the data

inTrain = createDataPartition(y=training$classe, p=0.6, list = FALSE)
subTraining = training[inTrain,]
subTest = training[-inTrain,]

## Getting and cleaning data

# Step 1 - cleaning variables

## Remove columns with more than 40% NA
subTraining = subTraining[, -which(colMeans(is.na(subTraining)) > 0.4)]
sum(is.na(subTraining)) # Note this actually removes all NAs

## Remove columns with near zero variance variables

nzv_cols = nearZeroVar(subTraining)
subTraining = subTraining[,-nzv_cols]

## Remove ID column, username and timestamps as it will confuse the ML algorithm and adds no information
## for this analysis

subTraining = subTraining[,-(1:5)]

## Now remove same columns from training and test data

cleanNames = names(subTraining)
subTest = subTest[cleanNames]
test = test[cleanNames[-54]] # already doesn't have last variable

# OK let's do some machine learning

## Boosting first

boostingModel = train(y = subTraining[,54], x = subTraining[,-54], method = 'gbm', verbose= FALSE)

boostingPredictions = predict(boostingModel, newdata=subTest)

confusionMatrix(subTest$classe, boostingPredictions)

## Now random forest

randomForestModel = train(y = subTraining[,54], x = subTraining[,-54], method = 'rf')

rfPredictions = predict(randomForestModel, newdata=subTest)

confusionMatrix(subTest$classe, rfPredictions)

# so let's use random forest for the test data

testPredictions = predict(randomForestModel, test)

