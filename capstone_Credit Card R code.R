
#################################
###  Capstone Project
#################################
##### Importing data
train1=read.csv("S:/capstone/fraudTrain.csv")
test1=read.csv("S:/capstone/fraudTest.csv")
#train=read.csv("C:/Users/sanit/OneDrive/Desktop/Capstone Project/fraudTrain_sample.csv")
#test=read.csv("C:/Users/sanit/OneDrive/Desktop/Capstone Project/fraudTest_sample.csv")

unsampled1=rbind(train1,test1)### merging test and train data to perform data understanding and EDA
dim(unsampled1)
##installing packages#########
library(gplots, warn.conflicts = FALSE)
library("ROCR")
library("PRROC")
library(dplyr) # for data manipulation
library(stringr) # for data manipulation
library(caret) # for sampling
library(ggplot2) # for data visualization
library(corrplot) # for correlations
library(Rtsne) # for tsne plotting
library(DMwR) # for smote implementation
library(ROSE)# for ROSE sampling
library(rpart)# for decision tree model
library(Rborist)# for random forest model

head(unsampled1)
colnames(unsampled1)
str(unsampled1)

########################EDA(Data analysis, bivariate analysis, univariate analysis,correlation) 
######and descriptive analysis)############################################

### checking missing values
a=sum(is.na(unsampled1$is_fraud))
a

### checking class imbalance
table(unsampled1$is_fraud)
### class imbalance in percentage
prop.table(table(unsampled1$is_fraud))

###Distribution of is_fraud labels

common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data = unsampled1, aes(x = factor(is_fraud), 
                      y = prop.table(stat(count)), fill = factor(is_fraud),
                      label = scales::percent(prop.table(stat(count))))) +
  geom_bar(position = "dodge") + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 3) + 
  scale_x_discrete(labels = c("no fraud", "fraud"))+
  scale_y_continuous(labels = scales::percent)+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Distribution of is_fraud labels") +
  common_theme

###pie chart
table(unsampled1$is_fraud)
unsampled1$is_fraud=ifelse(unsampled1$is_fraud== 1,"yes_fraud","no_fraud")
Fraud1=unsampled1$is_fraud
Fraud1=as.factor(unsampled1$is_fraud)
table(Fraud1)

freqfraud= table(Fraud1)
pie(freqfraud)
perce=round(freqfraud/463099*100)
perce
label=paste(names(freqfraud), perce, "%",sep=" ")
label
pie(freqfraud , main="fraud & no_fraud ratio", col= c(2,5), labels = label)
unsampled1$is_fraud=ifelse(unsampled1$is_fraud== "yes_fraud",1,0)
table(unsampled1$is_fraud)
###Distribution of Time & is_fraud

unsampled1$hour_of_day <- (unsampled1$unix_time/3600) %% 24 # Feature engineered - unix_time to hour_of_day
head(unsampled1)
unsampled1 %>%
  ggplot(aes(x = hour_of_day, fill = factor(is_fraud))) + geom_histogram(bins = 100)+
  labs(x = 'hour of day', y = 'No. of transactions') +
  ggtitle('Distribution of time of transaction by is_fraud') +
  facet_grid(is_fraud ~ ., scales = 'free_y') + common_theme

###Distribution of Amt & is_fraud
ggplot(unsampled1,aes(x=is_fraud, y=amt,group=is_fraud))+
  geom_boxplot()+
  ggtitle("Amt by is_fraud")

###Distribution of Amt & is_fraud=1
table(unsampled1$is_fraud)

unsampled1 %>%
  filter(is_fraud == "1") %>%
  ggplot(aes(x=is_fraud, y=amt,group=is_fraud))+
  geom_boxplot()+
  ggtitle("Amt by is_fraud")


###Distribution of gender & is_fraud (0,1)

ggplot(unsampled1, aes(is_fraud)) + geom_bar(aes(fill = gender), position = "dodge")

###Distribution of gender & is_fraud (1)
unsampled1 %>%
  filter(is_fraud == "1") %>%
  ggplot(aes(x=is_fraud)) + geom_bar(aes(fill = as.factor(gender)), position = "dodge")

#Distribution of category & is_fraud

unsampled1 %>%
  filter(is_fraud == "1") %>%
  ggplot(aes(x=is_fraud)) + geom_bar(aes(fill = as.factor(category)), position = "dodge")


###correlation plots
corelation<-cor(unsampled1[,sapply(unsampled1,is.numeric)],use="complete.obs",method="pearson")
library(corrplot)
corrplot(corelation, type = "lower", tl.col = 'black',  tl.srt = 20)
corrplot(corelation, method = 'number')

#####descriptive analysis
summary(unsampled1)


############### Data pre-processing (re-engineering, re-coding and removing unnecessary variables) 
##################to tackle imbalance problem and building models################################
##Data importing
train=read.csv("S:/capstone/fraudTrain_sample.csv")
test=read.csv("S:/capstone/fraudTest_sample.csv")
unsampled=rbind(train,test)### merging test and train data to perform data understanding and EDA
dim(unsampled)

#finding duplicate rows
nrow(unsampled[duplicated(unsampled), ])

###checkin for NULL values
a=sum(is.na(unsampled$is_fraud))
a


#Scaling numeric variables
unsampled$amt <- scale(unsampled$amt)
unsampled$merch_lat <- scale(unsampled$merch_lat)
unsampled$merch_long <- scale(unsampled$merch_long)
unsampled$lat <- scale(unsampled$lat)
unsampled$long <- scale(unsampled$long)
unsampled$city_pop <- scale(unsampled$city_pop)

##date and time extraction from transaction timestamp
unsampled$trans_date=as.Date(unsampled$trans_date_trans_time)
unsampled$trans_time=format(as.POSIXct(unsampled$trans_date_trans_time), format = "%H:%M")
unsampled$trans_hour=format(as.POSIXct(unsampled$trans_date_trans_time), format = "%H")
unsampled$trans_month=format((unsampled$trans_date),"%m") 
unsampled$trans_DD=format((unsampled$trans_date),"%d")
unsampled$trans_hour=as.numeric(unsampled$trans_hour)
unsampled$trans_month=as.numeric(unsampled$trans_month)
unsampled$trans_DD=as.numeric(unsampled$trans_DD)
unsampled$dob=as.Date(unsampled$dob)

head(unsampled)

###encoding gender to 1,0
unsampled$gender=ifelse(unsampled$gender=="M",1,0)


###Converting the categorical variables into Factor
unsampled$is_fraud <- as.factor(unsampled$is_fraud)
unsampled$zip <- as.factor(unsampled$zip)
unsampled$state <- as.factor(unsampled$state)
unsampled$gender <- as.factor(unsampled$gender)
unsampled$category <- as.factor(unsampled$category)
unsampled$trans_hour<-as.factor(unsampled$trans_hour)

unsampled$age = round(as.numeric(difftime(unsampled$trans_date,unsampled$dob))/365,0)
head(unsampled)
####eliminating unnecessary columns and keeping only required ones
colnames(unsampled)
data_cc=subset(unsampled,select=c(5,6,9,12,14,15,16,21,22,23,26,29))
colnames(data_cc)
head(data_cc)
#####splitting Train and Test
library(caTools)
set.seed(123)
split = sample.split(data_cc$is_fraud,SplitRatio=0.80)
train_data = subset(data_cc,split==TRUE)
test_data = subset(data_cc,split==FALSE)

dim(train_data)
dim(test_data)

######## Data imbalance problem ###############
################  measure of imbalance################ 
table(data_cc$is_fraud)
# class imbalance in percentage
prop.table(table(data_cc$is_fraud))


### Both- Over & Under Sampling
table(data_cc$is_fraud)
fraction_fraud_new <- 0.5
traindata_cc <- ROSE::ovun.sample(is_fraud~., train_data, method = "both", p= fraction_fraud_new, seed = 1234)$data
table(traindata_cc$is_fraud)
table(test_data$is_fraud)
str(traindata_cc)
dim(train_data)
dim(traindata_cc)
##########################################################################
############# Model Building #############################################
##########################################################################

####### 1st Model Fitting- Logistic Regression model#########

fit.lm=glm(is_fraud~., data = traindata_cc, family = binomial)
summary(fit.lm)

### Feature selection using AIC
library(MASS)
step=stepAIC(fit.lm)
colnames(traindata_cc)
colnames(test_data)

### prediction using glm model on test data
predict_lm <- predict(fit.lm, test_data, type = "response")
hist(predict_lm)
head(predict_lm)
table(test_data$is_fraud)
class(predict_lm)
str(predict_lm)
###########################################################################
###################  MODEL Evaluation######################################
###########################################################################

### ROC Curve
library(pROC)
roc1=roc(test_data[,10],predict_lm,plot=TRUE,legacy.axes=TRUE)

plot(roc1)
AUC_LM=roc1$auc
AUC_LM



### GLM evalaution- confusion matrix
library(caret)
pred_Y=ifelse(predict_lm > 0.9,1,0)
hist(pred_Y)
table(pred_Y)
roc=roc(test_data[,10],pred_Y,plot=TRUE,legacy.axes=TRUE)
roc$auc
confusionMatrix(as.factor(test_data[,10]), as.factor(pred_Y))


############# 2nd Model Fitting- Random Forest ######################
library(randomForest)
rfboth<-randomForest(as.factor(is_fraud) ~ ., data=traindata_cc,ntree=500, mtry=6)
rfboth

### confusion matrix
library(caret)
rf_predict <- predict(rfboth, test_data)
rf_cm<- confusionMatrix(data = rf_predict, test_data$is_fraud)
rf_cm
###########################################################################
###################  MODEL Evaluation######################################
###########################################################################
roc_RF=roc((test_data[,10]),as.numeric(rf_predict),plot=TRUE,legacy.axes=TRUE,col='red',main = "ROC curve for Random forest ")

auc_RF=as.numeric(roc_RF$auc)
auc_RF

table(test_data$is_fraud)
CM<-table(as.numeric(rf_predict),test_data[,10])
CM
TN =CM[1,1]
TN
TP =CM[2,2]
TP
FN =CM[1,2]
FN
FP =CM[2,1]
FP

precision_RF =(TP)/(TP+FP)
recall_RF =(TP)/(TP+FN)
f1_score_RF=2*((precision_RF*recall_RF)/(precision_RF+recall_RF))
accuracy_RF  =(TP+TN)/(TP+TN+FP+FN)
precision_RF
recall_RF
f1_score_RF
accuracy_RF

############# 3rd Model Fitting- Decision Tree ######################


library(rpart)
library(rpart.plot)
mod_tree = rpart(is_fraud ~ .,data = traindata_cc,method = "class")
rpart.plot(mod_tree,cex=0.45,extra = 0,type=5,box.palette = "BuRd")
pred_tree = predict(mod_tree,test_data,type="class")
roc_tree=roc.curve(test_data$is_fraud,pred_tree,plotit = TRUE,
                       col="red",main = "ROC curve for Decision Tree Algorithmn ",
                       col.main="darkred")
auc_tree=as.numeric(roc_tree$auc)
auc_tree
###########################################################################
###################  MODEL Evaluation######################################
###########################################################################
##confusion matrix
confusionMatrix(pred_tree,test_data[,10])
CM<-table(pred_tree,test_data[,10])
CM
TN =CM[1,1]
TN
TP =CM[2,2]
TP
FN =CM[1,2]
FN
FP =CM[2,1]
FP

precision_DT =(TP)/(TP+FP)
recall_DT =(TP)/(TP+FN)
f1_score_DT=2*((precision_DT*recall_DT)/(precision_DT+recall_DT))
accuracy_DT  =(TP+TN)/(TP+TN+FP+FN)
precision_DT
recall_DT
f1_score_DT
accuracy_DT







