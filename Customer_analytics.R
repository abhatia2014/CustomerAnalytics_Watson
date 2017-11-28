# link
# https://www.r-bloggers.com/customer-analytics-using-deep-learning-with-keras-to-predict-customer-churn/



# Load Libraries ----------------------------------------------------------

library(tidyverse)
library(keras)
library(lime)
library(tidyquant)
library(rsample)
library(recipes)
library(yardstick)
library(corrr)

# install keras if not installed previously

install_keras()


# Churn Modeling with Artificial Neural Networks (Keras) ------------------

# read in the dataset

churn=read_csv(file.choose())

glimpse(churn)


# Data Preprocessing ------------------------------------------------------

# using Recipes package

# remove unnecessary data
# removing customer ID variable
# dropping NA
# bringing Churn variable to front

churn_tbl=churn %>% 
  select(-customerID) %>% 
  drop_na() %>% 
  select(Churn,everything())

glimpse(churn_tbl)


# splitting into training and test sets

# using the package rsample

set.seed(100)

train_set=initial_split(churn_tbl,prop=0.8)

# we can retrieve our training and test set using training and testing functions

train_tbl=training(train_set)
test_tbl=testing(train_set)

# ANN work best when the data is one hot encoded, scaled and centered

# Tenure feature need to be discretize (converted to groups)

# we divide in 6 groups of 12 months duration each

ggplot(churn_tbl,aes(tenure))+geom_histogram(bins=30)+
  geom_text(stat='bin',aes(label=..count..))



ggplot(churn_tbl,aes(log(TotalCharges)))+geom_histogram(bins=30)

# to determine if the log transformation of total charges improves correlation

train_tbl %>% 
  select(Churn,TotalCharges) %>% 
  mutate(
    Churn=Churn %>% as.factor() %>% as.numeric(),
    LogTotalCharges=log(TotalCharges)
  ) %>% 
  correlate() %>% 
  focus(Churn) %>% 
  fashion()

# rowname Churn
# 1    TotalCharges  -.20
# 2 LogTotalCharges  -.25

# as correlation between Churn and LogTotalCharges is more than TotalCharges
# implies log transformation improves correlation

# one hot encoding and feature scaling

# converting categorical data into sparse data

# also called creating dummy variables

# perform preprocessing with the recipes package

# step 1 : create a recipe

# the step functions used here are
# 1. step_discretize to discretize tenure function
# 2. step_log to log transform 'Total Charges'
# 3. step_dummy to one-hot-encode categorical data
# 4. step_center to center data
# 5. step_scale to scale data

recipe_object=recipe(Churn~.,data = train_tbl) %>% 
  step_discretize(tenure,options = list(cuts=6)) %>% 
  step_log(TotalCharges) %>% 
  step_dummy(all_nominal(),-all_outcomes()) %>% 
  step_center(all_predictors(),-all_outcomes()) %>% 
  step_scale(all_predictors(),-all_outcomes()) %>% 
  prep(data=train_tbl)

recipe_object

# baking with the recipe object

# you can apply the recipe to any dataset with the bake() function

# we apply to training and test dataset to convert them to ML datasets

x_train_tbl= bake(recipe_object,newdata = train_tbl)
x_test_tbl=bake(recipe_object,newdata = test_tbl)

glimpse(x_train_tbl)

# store the response variable for training and test sets as separate variables

y_train_vec=ifelse(pull(train_tbl,Churn)=="Yes",1,0)
y_test_vec=ifelse(pull(test_tbl,Churn)=="Yes",1,0)


# Building the ANN Keras model --------------------------------------------

# remove the target variable from training and test set

x_train_tbl$Churn=NULL
x_test_tbl$Churn=NULL

model_keras=keras_model_sequential() # to initialize the keras model


model_keras %>% 
  # first hidden layer
  layer_dense(
    units = 16,
    kernel_initializer = 'uniform',
    activation = 'relu',
    input_shape = ncol(x_train_tbl)) %>% 
   # drop out to prevent overfitting
  layer_dropout(rate = 0.1) %>% 
  # second hidden layer
  layer_dense(
    units = 16,
    kernel_initializer = 'uniform',
    activation = 'relu') %>% 
  #dropout to prevent overfitting
  layer_dropout(rate=0.1) %>% 
  #outout layer
  layer_dense(
    units = 1,
    kernel_initializer = 'uniform',
    activation = 'sigmoid') %>% 
  # complile ANN
  compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics=c('accuracy')
  )


model_keras

# we use the fit() function to run the ANN model on the training dataset

fit_keras=fit(
  object = model_keras,
  x=as.matrix(x_train_tbl),
  y = y_train_vec,
  batch_size = 50,
  epochs = 35,
  validation_split = 0.30 # 30% data used for validation
)

fit_keras

# plot the results separately of the training/validation history of keras model

plot(fit_keras)+
  theme_tq()+
  scale_color_tq()+
  scale_fill_tq()+
  labs(title="Deep Learning Training Results")


# Making Predictions ------------------------------------------------------


# predictions on the unseen test set

# predicted class

predict_keras_class=predict_classes(object = model_keras,
                                    x = as.matrix(x_test_tbl)) %>% 
  as.vector()

predict_keras_class_prob=predict_proba(object = model_keras,
                                       x = as.matrix(x_test_tbl)) %>% 
  as.vector()


# Inspect performance with Yardstick package ------------------------------

# we'll use fct_recode() from forcats package to assist recodeing yes/no

# format test data and predictions for yardstick package
library(forcats)

estimate_keras_perf=tibble(
  truth=as.factor(y_test_vec) %>% 
    fct_recode(yes='1',no='0'),
  estimate=as.factor(predict_keras_class) %>% 
    fct_recode(yes="1",no='0'),
  class_prob=predict_keras_class_prob
)

estimate_keras_perf

library(caret)

confusionMatrix(estimate_keras_perf$estimate,estimate_keras_perf$truth)

# remove caret package
detach("package:caret", unload=TRUE)

# confusion table using keras

estimate_keras_perf %>% 
  conf_mat(truth,estimate)

# accuracy

estimate_keras_perf %>% metrics(truth,estimate)

# AUC

estimate_keras_perf %>% roc_auc(truth,class_prob)

# precision and recall

estimate_keras_perf %>% precision(truth,estimate)

estimate_keras_perf %>% recall(truth,estimate)

# F1 score

estimate_keras_perf %>% f_meas(truth,estimate,beta=1)


# Checking performance with other H2O based models using MLR --------------

mlr_train=x_train_tbl
mlr_train$Churn=train_tbl$Churn

mlr_test=x_test_tbl
mlr_test$Churn=test_tbl$Churn

library(mlr)

# build a training task

train_task=makeClassifTask(data = mlr_train,target = "Churn")

# define a learner

alllearners=listLearners(train_task)

# select learner

train_learn=makeLearner(cl = "classif.h2o.gbm",predict.type = "prob",
                        fix.factors.prediction = TRUE)

# set resampling strategy

train_resample=makeResampleDesc(method ="CV",iters=3 ,stratify = TRUE)

# set accuracy measures
listMeasures(train_task)

train_measures=list(acc,auc)

# perform resample 

train_resample_gbm=resample(learner = train_learn,task = train_task,
                            resampling = train_resample,measures = train_measures)

# train using mlr

final_model_h2o.gbm=mlr::train(learner = train_learn,task = train_task)


# test on test set

predictions_test_h2o.gbm=predict(object = final_model_h2o.gbm,newdata = mlr_test)

library(caret)
confusionMatrix(predictions_test_h2o.gbm$data$response,
                predictions_test_h2o.gbm$data$truth)

performance(pred = predictions_test_h2o.gbm,measures = list(acc,auc,f1))

# Model explanation with Lime ---------------------------------------------


