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
