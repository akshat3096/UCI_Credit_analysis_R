#reading in data
UCI.Credit.Card <- read.csv("~/Eg_Datasets/UCI-Credit-Card.csv", header=T)
str(UCI.Credit.Card)
summary(UCI.Credit.Card)

#proportion of classes
prop.table(table(UCI.Credit.Card$default.payment.next.month))

#standardise the data(although not necessary in classification tasks)
standardise <- function(x){
  (x-min(x))/(max(x)-min(x))
}
UCI.Credit.Card <- as.data.frame(lapply(UCI.Credit.Card, standardise ))
summary(UCI.Credit.Card)

#data partition
library(caret)
set.seed(99)
par <- createDataPartition(UCI.Credit.Card$default.payment.next.month,p=0.75,list=F)
train <- UCI.Credit.Card[par,]
test <- UCI.Credit.Card[-par,]

#selecting a logistic regression model
library(MASS)

#stepwise logistic regression
full<- glm(default.payment.next.month~.,data=train,family = binomial)
null <- glm(default.payment.next.month~1,data = train,family = binomial)
step(null,scope=list(lower=null,upper=full),direction = "both")

#final model selected on the basis of lowest AIC value
fit1 <-glm(formula = default.payment.next.month ~ PAY_0 + LIMIT_BAL + 
             PAY_AMT1 + PAY_3 + BILL_AMT1 + AGE + PAY_AMT2 + BILL_AMT2 + 
             EDUCATION + PAY_2 + MARRIAGE + PAY_5 + SEX + PAY_AMT4 + ID + 
             PAY_AMT6, family = binomial, data = train)
summary(fit1)

#confusion matrix 
p <- predict(fit1,newdata = test,type = "response")
head(p)
pred1 <- ifelse(p>0.5,1,0)
tab <- table(pred1,test$default.payment.next.month)
tab
sum(diag(tab))/sum(tab) #accuracy 80.6%

## Trying to boost the accuracy

## Using cross validation#
##########################
control <- trainControl(method = "cv", number=10)
fit2 <- train(as.factor(default.payment.next.month) ~ PAY_0 + LIMIT_BAL + 
                PAY_AMT1 + PAY_3 + BILL_AMT1 + AGE + PAY_AMT2 + BILL_AMT2 + 
                EDUCATION + PAY_2 + MARRIAGE + PAY_5 + SEX + PAY_AMT4 + ID + 
                PAY_AMT6 ,data=train,trControl=control,method="glm",family="binomial")
summary(fit2)

#prediction
p <- predict(fit2,newdata = test)

#performance on the test data (Accuracy=80.6%)
confusionMatrix(p,as.factor(test$default.payment.next.month))
varImp(fit2) #list of importance of variables 
#variables with 0 importance can be removed 

## Using Gradient Boosting#
###########################
fit3 <- train(as.factor(default.payment.next.month)~PAY_0 + LIMIT_BAL + 
                PAY_AMT1 + PAY_3 + BILL_AMT1 + AGE + PAY_AMT2 + BILL_AMT2 + 
                EDUCATION + PAY_2 + MARRIAGE + PAY_5 + SEX + PAY_AMT4 + ID + 
                PAY_AMT6 ,data=train,trControl=control,method="gbm")
#prediction 
p1 <- predict(fit3,newdata = test)
#performance on the test data
confusionMatrix(p1,as.factor(test$default.payment.next.month)) #(accuracy 81.6%)


## Using random forest#
#######################

#without tuning and specifying mtry
fit3 <- train(as.factor(default.payment.next.month) ~ PAY_0 + LIMIT_BAL + 
                PAY_AMT1 + PAY_3 + BILL_AMT1 + AGE + PAY_AMT2 + BILL_AMT2 + 
                EDUCATION + PAY_2 + MARRIAGE + PAY_5 + SEX + PAY_AMT4 + ID + 
                PAY_AMT6 ,data=train,trControl=control,method="rf",verboseIter = TRUE)

#prediction 
p1 <- predict(fit3,newdata = test) 
#performance on the test data
confusionMatrix(p1,as.factor(test$default.payment.next.month))  #(accuracy 82% approx)

##using neural networks##
#########################
modelNnet = train(as.factor(default.payment.next.month) ~ PAY_0 + LIMIT_BAL + 
                    PAY_AMT1 + PAY_3 + BILL_AMT1 + AGE + PAY_AMT2 + BILL_AMT2 + 
                    EDUCATION + PAY_2 + MARRIAGE + PAY_5 + SEX + PAY_AMT4 + ID + 
                    PAY_AMT6,data=train,
                  method="nnet",
                  preProcess=c('center', 'scale'),
                  trace=FALSE,
                  tuneLength=10,
                  trControl=control)
#prediction
predNnet= predict(modelNnet, test)
confusionMatrix(predNnet,as.factor(test$default.payment.next.month))  #(accuracy 82% approx)
