# Code for mushroom classification project 
# This code prints out accuracy data frame for all 5 models
# This code prints out top 5 features for all 5 models
# This code also generates 5 models in the Workspace
# Note: model3 is in rf_rfe$fit
# Note: this code will take a while to finish 
# Note: the code for rmarkdown kable table can be found in the Mushroom_Project.Rmd file 
# Note: the code for data visualization plots can be found in the Mushroom_Project.Rmd file

# load required libraries for importing data and data wrangling

library(tidyverse)
library(caret)

# import data from UCI website 
url_dat <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data' 

# column names from the agaricus-lepiota.names file from the UCI data website, 
# https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/
header <- c('class', 'cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment',
            'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root',
            'stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
            'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type',
            'spore-print-color','population','habitat')%>%
  str_replace_all('-','_')  # replace '-' with '_' for better wrangling 
# import mushroom data into a data.frame with header as column names 
# poisnous mushroom will be level 1 as positive class during confusion matrix
dat <- read.csv(url_dat,header = FALSE,col.names = header)%>%
  mutate(class=factor(class,levels=c('p','e')))  

# split data set into test(20%) and train(80%) sets
# set seed for reproducibility
set.seed(1)
# 20% test and 80% train 
test_index <- createDataPartition(y = dat$class,times = 1,p = 0.2,list = FALSE)  
test <- dat[test_index,]
train <- dat[-test_index,]           
# set y as the class labels for mushrooms in the test set 
y <- test$class
# remove class labels from the test set
test <- test%>%select(-class)
# remove unncessary data
rm(url_dat,header,dat,test_index)

# remove veil type, odor, population, and habitat 
#(reasons can be found in Mushroom_Project PDF report)
train <- train%>%select(-veil_type,-odor,-population,-habitat)

# model1: decision tree 
# load libraries for decision tree
library(rpart)
library(rpart.plot)
# fit model_1 using train set, use rpart.control to find the best cp (set cp to zero first)
model_1 <- rpart(class~., data = train,method = 'class', control = rpart.control(cp=0))
plotcp(model_1)  # visually look at the best cp 
opt_index <- which.min(model_1$cptable[,'xerror'])  # find the index for the lowest xerror
cp_opt <- model_1$cptable[opt_index,'CP']  # use opt_index to find the optimal cp 
model_1 <- prune(tree = model_1,cp=cp_opt)  # prune the tree with the best cp
# plot the tree, set poisonous as red and edible as green
rpart.plot(x = model_1, box.palette = c('red','green'), type =5, extra = 0)
# use the varImp function from caret package to plot the importance levels from the most to least important 
varImp(model_1)%>%
  mutate(features = rownames(.))%>%
  ggplot(aes(x=reorder(features, Overall),y = Overall))+
  geom_bar(stat = 'identity',aes(fill=features))+
  theme(axis.text.y=element_text(hjust=1,face ='bold'), legend.position = 'none')+
  xlab('')+
  ylab('Importance Level')+
  coord_flip()
# save the top 5 most important features in the character vector top_5_1 for model comparison
top_5_1 <- varImp(model_1)%>%
  mutate(var=rownames(.))%>%
  arrange(desc(Overall))%>%
  head(n=5)%>%.$var
# test the model accuracy on test set
# transform the test set by removing the veil type, odor, population, and habitat
test_m<- test%>%
  select(-c('veil_type','habitat','odor','population'))
y_hat1 <- predict(model_1, newdata = test_m, type = 'class')
# save the model_1 accuracy result as accuracy_1 for final model comparison table
accuracy_1 <- paste0(confusionMatrix(y_hat1,y)$overall['Accuracy']*100, '%')
# remove unncessary data
rm(test,opt_index,cp_opt, y_hat1)

# model2: graident boosted decision tree (GBM) 
# load the library for gbm tree
library(gbm)
# for binary classification, convert 'p' as 1 and 'e' as 1, p is positive class in confusion matrix 
train_m <- train%>%mutate(class=ifelse(class =='p',1,0))
set.seed(20) # set a seed for reproduciblity
# fit model_2 using train set, use gbm function
# distribution for binary classification is bernoulli distribution (1 or 0 outcome)
# use cross validation of 5 folds and set n.tree to be 10000 (iterations)
model_2<- gbm(class ~., distribution = 'bernoulli',data = train_m, n.trees = 10000,cv.folds = 5)  
# use gbm.perf function to find the optimal number of trees for prediction to prevent over fitting using a 5 fold cross validation 
ntree_opt<- gbm.perf(model_2, method = 'cv')
# summary.gbm function provides information for feature importance level
# the function automatically plots the default variable importance graph so it is important to set plotit to FALSE for customized plot 
summary.gbm(model_2, plotit = FALSE)%>%
  ggplot(aes(x=reorder(var, rel.inf),y = rel.inf))+
  geom_bar(stat = 'identity',aes(fill=var))+
  theme(axis.text.y=element_text(hjust=1,face ='bold'), legend.position = 'none')+
  xlab('')+
  ylab('Importance Level')+
  coord_flip()
# save the top 5 most important features in the character vector top_5_2 for model comparison
top_5_2 <- summary.gbm(model_2, plotit = FALSE)%>%top_n(n=5)%>%
  mutate(var=as.character(var))%>%.$var
# test the model accuracy on the transformed test set, test_m (see model1 decision tree section)
y_hat2<-predict(model_2, newdata = test_m,n.trees = ntree_opt, type = 'response')
# convert the predicted value back to poisonous (>0.5) and edible class factors
y_hat2 <- ifelse(y_hat2 >0.5, 'p','e')%>%factor(levels=c('p','e')) 
# save the model_2 accuracy result as accuracy_2 for final model comparison table
accuracy_2 <- paste0(confusionMatrix(y_hat2,y)$overall['Accuracy']*100, '%')
# remove unncessary data
rm(ntree_opt, y_hat2)

# model3: random forest with recursive feature elimination (RF_RFE) from caret package
# set a seed for reproduciblity
set.seed(10)
# use rfeControl function from caret package, rfFuncs is a predefined caret RFE function for random forest model 
# use cross validation of 10 folds
control <- rfeControl(functions = rfFuncs, method = 'cv',number = 10)
# rfe function performs recursive feature elimination, set the optimal size to be between 1 to 16 features 
rf_rfe <- rfe(x = train[,-1], y = train$class, sizes =1:16, rfeControl=control)  
# plot the variables results to see the optimal number of features 
plot(rf_rfe, type=c('g','o'),col='red')
# use the varImp function from caret package to plot the importance levels from the most to least important 
# rf_rfe$fit is the random forest model using the optimal number and combination of the features 
varImp(rf_rfe$fit)%>%mutate(features = rownames(.))%>%
  ggplot(aes(x=reorder(features, p),y = p))+
  geom_bar(stat = 'identity',aes(fill=features))+
  theme(axis.text.y=element_text(hjust=1,face ='bold'), legend.position = 'none')+
  xlab('')+
  ylab('Importance Level')+
  coord_flip()
# save the top 5 most important features in the character vector top_5_3 for model comparison
top_5_3 <- varImp(rf_rfe$fit)%>%
  mutate(var = rownames(.))%>%
  arrange(desc(p))%>%
  head(n=5)%>%.$var
# test the model accuracy on the transformed test set, test_m (see model1 decision tree section)
y_hat3 <- predict(rf_rfe$fit, newdata = test_m, type = 'class')
# save the model_3 accuracy result as accuracy_3 for final model comparison table
accuracy_3 <- paste0(confusionMatrix(y_hat3,y)$overall['Accuracy']*100, '%')
# remove unncessary data
rm(control, y_hat3)

# model4: decision tree using the RF_RFE selected features
# transform the train set by selecting only the RF_RFE selected features (saved in rf_rfe$optVariables as a character vector) 
# and the class label 
train_m <- train%>%
  select(class, rf_rfe$optVariables)
# transform the test_m set by selecting only the RF_RFE selected features (saved in rf_rfe$optVariables as a character vector)
test_m <- test_m%>%
  select(rf_rfe$optVariables)
# fit model_1_r using train_m and selected features, use rpart.control to find the best cp (set cp to zero first)
model_1_r <- rpart(class~., data = train_m,method = 'class', control = rpart.control(cp=0))
plotcp(model_1_r)  # visually look at the best cp 
opt_index_2 <- which.min(model_1_r$cptable[,'xerror'])  # find the index for the lowest xerror
cp_opt_2 <- model_1_r$cptable[opt_index_2,'CP']  # use opt_index_2 to find the optimal cp
model_1_r <- prune(tree = model_1_r,cp=cp_opt_2)  # prune the tree with the best cp 
# plot the tree, set poisonous as red and edible as green
rpart.plot(x = model_1_r, box.palette = c('red','green'), type =5, extra = 0)
# use the varImp function from caret package to plot the importance levels from the most to least important 
varImp(model_1_r)%>%
  mutate(features = rownames(.))%>%
  ggplot(aes(x=reorder(features, Overall),y = Overall))+
  geom_bar(stat = 'identity',aes(fill=features))+
  theme(axis.text.y=element_text(hjust=1,face ='bold'), legend.position = 'none')+
  xlab('')+
  ylab('Importance Level')+
  coord_flip()
# save the top 5 most important features in the character vector top_5_4 for model comparison
top_5_4 <- varImp(model_1_r)%>%
  mutate(var=rownames(.))%>%
  arrange(desc(Overall))%>%
  head(n=5)%>%.$var
# test the model accuracy on the new transformed test set, test_m
y_hat4 <- predict(model_1_r, newdata = test_m, type = 'class')
# save the model_1_r accuracy result as accuracy_4 for final model comparison table
accuracy_4 <- paste0(confusionMatrix(y_hat4,y)$overall['Accuracy']*100, '%')
# remove unncessary data
rm(train,opt_index_2,cp_opt_2, y_hat4)

# model5: gradient boosted decision tree (GBM) using the RF_RFE selected features
# for binary classification, convert 'p' as 1 and 'e' as 1, p is positive class in confusion matrix 
train_m <- train_m%>%mutate(class=ifelse(class =='p',1,0))  
set.seed(20) # set a seed for reproduciblity
# fit model_2_r using train_m (see model_4 decision tree revisit section), use gbm function
# distribution for binary classification is bernoulli distribution (1 or 0 outcome)
# use cross validation of 5 folds and set n.tree to be 10000 (iterations)
model_2_r <- gbm(class ~., distribution = 'bernoulli',data = train_m, n.trees = 10000,cv.folds = 5)
# use gbm.perf function to find the optimal number of trees for prediction to prevent over fitting using a 5 fold cross validation 
ntree_opt_2<- gbm.perf(model_2_r, method = 'cv')  
# summary.gbm function provides information for feature importance level
# the function automatically plots the default variable importance graph so it is important to set plotit to FALSE for customized plot 
summary.gbm(model_2_r, plotit = FALSE)%>%
  ggplot(aes(x=reorder(var, rel.inf),y = rel.inf))+
  geom_bar(stat = 'identity',aes(fill=var))+
  theme(axis.text.y=element_text(hjust=1,face ='bold'), legend.position = 'none')+
  xlab('')+
  ylab('Importance Level')+
  coord_flip()
# save the top 5 most important features in the character vector top_5_5 for model comparison
top_5_5 <- summary.gbm(model_2_r, plotit = FALSE)%>%
  top_n(n=5)%>%
  mutate(var=as.character(var))%>%.$var
# test the model accuracy on the transformed test set, test_m (see model_4 decision tree revisit section)
y_hat5<-predict(model_2_r, newdata = test_m,n.trees = ntree_opt_2, type = 'response')
# convert the predicted value back to poisonous (>0.5) and edible class factors
y_hat5<- ifelse(y_hat5 >0.5, 'p','e')%>%factor(levels=c('p','e'))
# save the model_2_r accuracy result as accuracy_5 for final model comparison table
accuracy_5<- paste0(confusionMatrix(y_hat5,y)$overall['Accuracy']*100, '%')
# remove unncessary data
rm(train_m, test_m, ntree_opt_2, y_hat5, y)

# display the 5 model results 

# Accuracy data frame 
# column 1, the 5 models
# model 2, the accuracy results
print(data.frame(models = c('tree','GBM','RF_RFE','tree revisit','GBM revisit'),
           accuracy = c(accuracy_1, accuracy_2, accuracy_3, accuracy_4, accuracy_5)))
# remove unncessary data
rm(accuracy_1,accuracy_2,accuracy_3, accuracy_4,accuracy_5)

# top 5 feature data frame 
# each column represents each model 
# the most important feature is listed in row 1 
# the 5th most important feature is listed in row 5
print(data.frame(tree_top_5 = top_5_1, GBM_top_5 = top_5_2,
           RF_RFE_top_5 = top_5_3, tree_top_5_r = top_5_4, 
           GBM_top_5_r = top_5_5))
# remove unncessary data
rm(top_5_1,top_5_2,top_5_3,top_5_4,top_5_5)
