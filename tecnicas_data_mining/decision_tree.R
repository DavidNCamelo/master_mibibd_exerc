# Now with the prepared data
# we can create a decision tree model

# Required library
library(rpart)
library(ROCR)

# load data
source('diabetes_data.R')

# Getting data
data_list <- load_diabetes_data()

target <- data_list$target

predictores <- names(data_list$predictors)

train_set <- data_list$train[, c(predictores, target)]

# decision tree
#dt <- rpart(
#  formula = as.formula(paste(data_list$target, "~ .")),
#  data = data_list$train,
#  method = "class",
#  parms = list(split = "gini"),
#  control = list(minsplit = 20, cp = 1e-12)
#)

dt2 <- rpart(
  formula = as.formula(paste(target, "~ .")),
  data = train_set,
  method = "class",
  parms = list(split = "gini"),
  control = list(minsplit = 20, cp = 1e-12)
)


dt2$cptable

cp_min <- dt2$cptable[which.min(dt2$cptable[, "xerror"]), "CP"]

plotcp(dt2)

rpart.plot::prp(dt2)

# Prune tree

prune_dt <- prune(dt2, cp = cp_min)
plotcp(prune_dt)

rpart.plot::prp(prune_dt)


# Now it's ready to evaluate quality by ROC and LIFT
# First it's necessaty to treain and test the final model
prune_dt_train <- predict(prune_dt)[, 2]

prune_dt_test <- predict(
  prune_dt,
  newdata = data_list$test[, c(predictores, target)],
  type = 'prob'
)[, 2]

# Evaluating predictors
train_pred <- prediction(prune_dt_train, data_list$train[[target]])
test_pred <- prediction(prune_dt_test, data_list$test[[target]])

# Render True positive rate plot
plot(performance(train_pred, "tpr", "rpp"), col = "blue", lwd = 2)

plot(performance(test_pred, "tpr", "rpp"), col = "red", lwd = 2, add = TRUE)

abline(a = 0, b = 1)

legend(
  "bottomright",
  c("Train", "Test", "Random"),
  col = c("blue", "red", "black"),
  lwd = c(2, 2, 2),
  title = "Tree Model"
)
grid()


# Lift graph
plot(performance(train_pred, "lift", "rpp"), col = "blue", lwd = 2)

plot(
  performance(test_pred, "lift", "rpp"),
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
  title = "Tree Model"
)

grid()


# ROC plot
plot(performance(train_pred, "tpr", "fpr"), col = "blue", lwd = 2)

plot(
  performance(test_pred, "tpr", "fpr"),
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
  title = "Tree Model"
)

grid()

# AUC

performance(train_pred, "auc")@y.values[[1]]

performance(test_pred, "auc")@y.values[[1]]
