# data-science
Code for Practical Data Science with R

## Final Assignment
library(caret)
library(MASS)
# install.packages("e1071")
# install.packages("phonTools")
# install.packages("naivebayes")
library(naivebayes)
library(e1071)
library(phonTools)

# Data import and management

carriers <- read.table("carriers.txt", header = T)
normals <- read.table("normals.txt", header = T)
hp <- read.table("human-phage.txt")

biomed <- rbind(carriers, normals)

indic <- zeros(194, 1)
indic[1:67, 1] = 1

biomed <- cbind(biomed, indic)

# Analysis of biomedical data

## PCA

bio.pc <- prcomp(biomed[, 1:6], scale = T)
# scaled as otherwise 'date' dominates
biplot(bio.pc)
# looking at the biplot, we can see that the date the test
# was taken is not as important as the other measurements. 

bio.pc$rotation
# From the rotation matrix we can see that the variables that
# contribute most to the first PC are m1, m3 and m4.
# Therefore we will make a scores plot to look at the PCs.

groups <- as.factor(biomed$indic)

for(i in 1:4){
  for(j in 1:4){
    plot(bio.pc$x[, i], bio.pc$x[, j], col = groups, main = paste("PC", toString(i),"against PC", toString(j))) 
  }
}

# The best separation seems to be between PC1 and PC4.
bio.pc$rotation
# This tells us that the variables which matter most
# seem to be m2 and age.

summary(bio.pc)

## LDA

n <- nrow(biomed)
trainIndex <- sample(1:194, size = round(0.7 * n), replace = F)

train <- biomed[trainIndex, ]
test <- biomed[-trainIndex, ]

bio.ld <- lda(train[, 1:6], train[, 7])
bio.ld.pred <- predict(bio.ld, test[, 1:6])
table(predicted = bio.ld.pred$class, test[, 7])

# Or, with k-fold cross validation

train_control <- trainControl(method = "cv", number = 10, savePredictions = T)
bio.ldcv <- train(as.factor(indic)~., data = biomed, method = "lda", metric = "Accuracy", trControl = train_control, preProces = c("center", "scale"))

bio.ldcv$results

# Or, with LOOCV

bio.loocv <- lda(biomed[, 1:6], biomed[, 7], CV = T)
loocv.res <- table(biomed[, 7], bio.loocv$class)
loocv.res
diag(prop.table(loocv.res, 1))
sum(diag(prop.table(loocv.res)))

## Naive-Bayes

bio.nb <- train(as.factor(indic)~., method = "naive_bayes", data = biomed, trControl = trainControl(method = "LOOCV"), preProcess = c("center", "scale"))
bio.nba <- naive_bayes(train[, 1:6], train[, 7])

bio.nb$results
bio.nb.pred <- predict(bio.nba, test[, 1:6])

## ROC for different models

real = test[, 7]
table(true = real, predicted = bio.ld.pred$class)
table(true = real, predicted = bio.nb.pred)

# We can already see that Naive-Bayes performs better than
# LDA.

