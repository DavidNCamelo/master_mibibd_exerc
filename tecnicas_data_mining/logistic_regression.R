# Now with the prepared data
# we can create a logistic regression model

# Required library
library(caret)
library(MASS)
library(ROCR)

# load data
source('diabetes_data.R')

# Getting data
data_list <- load_diabetes_data()

# Import variables
target <- data_list$target
train_set <- data_list$train
test_set <- data_list$test


# Model
log_model <- glm(
  formula = as.formula(paste(target, "~ .")),
  data = train_set,
  family = binomial(link = "logit")
)

# Summary model
summary(log_model)


# To have a good varaibles selection
# will be implemented Stepwise metodology
new_log_model <- stepAIC(log_model, trace = TRUE)

summary(new_log_model)

confint.default(new_log_model)


# New it's ready to evaluate quality

# Full model
log_model_train <- predict(log_model, type = 'response')

log_model_test <- predict(log_model, newdata = test_set, type = 'response')


# Final model
new_log_model_train <- predict(new_log_model, type = 'response')

new_log_model_test <- predict(
  new_log_model,
  newdata = test_set,
  type = 'response'
)


# Just focusing on the final model
log_train_pred <- prediction(new_log_model_train, data_list$train[[target]])
log_test_pred <- prediction(new_log_model_test, data_list$test[[target]])


# true positive rate
plot(performance(log_train_pred, "tpr", "rpp"), col = "blue", lwd = 2)

plot(
  performance(log_test_pred, "tpr", "rpp"),
  col = "red",
  lwd = 2,
  add = TRUE
)

abline(a = 0, b = 1)

legend(
  "bottomright",
  c("Train", "Test", "Random"),
  col = c("blue", "red", "black"),
  lwd = c(2, 2, 2),
  title = "Logistic Model"
)

grid()

# Lift graph
plot(performance(log_train_pred, "lift", "rpp"), col = "blue", lwd = 2)

plot(
  performance(log_test_pred, "lift", "rpp"),
  col = "red",
  lwd = 2,
  add = TRUE
)

abline(a = 1, b = 0)

legend(
  "topright",
  c("Train", "Test", "Random"),
  col = c("blue", "red", "black"),
  lwd = c(2, 2, 2),
  title = "Logistic Model"
)

grid()

# ROC plot
plot(performance(log_train_pred, "tpr", "fpr"), col = "blue", lwd = 2)

plot(
  performance(log_test_pred, "tpr", "fpr"),
  col = "red",
  lwd = 2,
  add = TRUE
)

abline(a = 0, b = 1)

legend(
  "bottomright",
  c("Train", "Test", "Random"),
  col = c("blue", "red", "black"),
  lwd = c(2, 2, 2),
  title = "Logistic Model"
)

grid()

# AUC

performance(log_train_pred, "auc")@y.values[[1]]

performance(log_test_pred, "auc")@y.values[[1]]
