# clear all variables
rm(list = ls(all = TRUE))
graphics.off()

# install and load packages
libraries = c("caret","LogicReg","e1071","pROC","MASS","ROCit")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# set the working directory
setwd("~/Desktop/IRTG/Presentations/SVM")

# Read in data
data = read.csv("p2p.csv")
data = data[-1]
data$status = factor(data$status)

# Split into training and test data
set.seed(1234)
ind = createDataPartition(y=data$status, p=0.75, list=F)
train = data[ind,]
test = data[-ind,]

# Step AIC
fit1 = glm(status ~ ., data=train,family=binomial())
summary(fit1)
step = stepAIC(fit1, direction  = "both")
step$anova

# Logistic regression
fit_log = glm(status~ratio003 + ratio004 + ratio005 + ratio006 + 
                ratio011 + ratio012 + DPO + DSO + turnover + ratio036 + ratio037 + 
                ratio039 + ratio040,family="binomial",data=train)
pre_log = as.numeric(predict(fit_log,newdata=test,type="response"))
class_log = factor(ifelse(pre_log>0.5,1,0))

# SVM
ctrl = trainControl(classProbs = T,method = "repeatedcv", number = 10, repeats = 1)
fit_svm_prob = train(status~ratio003 + ratio004 + ratio005 + ratio006 + 
                       ratio011 + ratio012 + DPO + DSO + turnover + ratio036 + ratio037 + 
                       ratio039 + ratio040,data=train,method="svmRadial",trConrol=ctrl,prob.model=TRUE)
fit_svm_class = train(status~ratio003 + ratio004 + ratio005 + ratio006 + 
                        ratio011 + ratio012 + DPO + DSO + turnover + ratio036 + ratio037 + 
                        ratio039 + ratio040,data=train,method="svmPoly",trConrol=ctrl)
class_svm = predict(fit_svm_class,newdata=test)
pre_svm = predict(fit_svm_prob,newdata=test,type="prob")

# Confusion matrix
confusionMatrix(class_svm,test$status)
confusionMatrix(class_log,test$status)

# ROC curve
jpeg("logit.jpg",width=600,height=600)
plot(rocit(pre_log,as.numeric(test$status)),YIndex=F)
dev.off()
jpeg("svm.jpg",wigth=600,height=600)
plot(rocit(pre_svm[,2],as.numeric(test$status)),YIndex=F)
dev.off()

# AUC
rocit(pre_log,as.numeric(test$status))$AUC
rocit(pre_svm[,2],as.numeric(test$status))$AUC

# Brier Score
mean((pre_log-(as.numeric(test$status)-1))^2)
mean((pre_svm[,2]-(as.numeric(test$status)-1))^2)