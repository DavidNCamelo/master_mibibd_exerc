# Required libraries
library(dplyr)
library(ggplot2)
library(rpart)
library(nnet)
library(caret)


# Defining data distribution
f <- function(x) {
  if (x < 0) {
    return(x * sin(1 / x))
  } else if (x <= 40) {
    return(sin(x / 40 * 5 * pi))
  } else {
    return(-log(x - 39))
  }
}

# Defining dataset
x <- c(
  seq(-30, 0, length.out = 50),
  seq(-5, 0, length.out = 100),
  seq(0, 40, length.out = 500),
  seq(40, 100, length.out = 50)
)
y <- sapply(x, f) + rnorm(x, sd = 0.7)
# by dataframe
set.seed(308)
data = data.frame(x, y)[sample(length(x)), ]

# Plotting
puntos <- ggplot(data) +
  geom_point(aes(x = x, y = y), color = "indianred3") +
  geom_line(aes(x = x, y = sapply(x, f)), color = "black", linewidth = 1)
puntos


# 1. Creating train and test samples
# To do it, create a split var to separate values
# the rules are 50% for each one
set.seed(123)
split <- round(nrow(data) * 0.5)
train_data <- (1:nrow(data) <= split)
test_data <- !train_data

# Both will be necessary in a datarframe too
train <- data[train_data, ]
test <- data[test_data, ]

# 2. Adjusting linear model
linear_model <- lm(y ~ x, data = train)
summary(linear_model)


# Plot Linear model
ggplot(train, aes(x = x, y = y)) +
  geom_point(color = "indianred3") +
  geom_smooth(method = "lm", se = FALSE, color = "forestgreen", linewidth = 1)

# Predicting test
linear_pred <- predict(linear_model, data)
data$linear_pred <- linear_pred
linear_pred_test <- predict(linear_model, test)
# Lineal error
linear_error <- linear_pred_test - data$y

# Visualize linear error
ggplot(data, aes(colour = train_data)) +
  geom_point(aes(x = linear_pred, y = linear_error)) +
  geom_smooth(aes(x = linear_pred, y = linear_error), method = 'loess')

# Evaluating
train_idx <- 1:split
test_idx <- (split + 1):nrow(data)

linear_mse_train <- mean((data$y[train_idx] - linear_pred[train_idx])^2)
linear_mse_test <- mean((data$y[test_idx] - linear_pred[test_idx])^2)

# R^2 = 1 - SSE/SST
linear_SSE <- sum((test$y - linear_pred_test)^2)
linear_SST <- sum((test$y - mean(test$y))^2)

linear_R2_test <- 1 - linear_SSE / linear_SST
linear_R2_test


# 3. Adjusting Decision Tree
set.seed(123)
tree_model <- rpart(y ~ x, data = train, method = 'anova', cp = 0.01)
rattle::fancyRpartPlot(tree_model)
summary(tree_model)

# Predict
tree_pred_test <- predict(tree_model, test)
tree_pred_train <- predict(tree_model, train)
#data$tree_pred <- tree_pred

test_tree_error <- tree_pred_test - data$y
train_tree_error <- tree_pred_train - data$y

# Main fuuctional
# Predict
tree_pred <- predict(tree_model, data)

tree_pred_test <- predict(tree_model, test)
data$tree_pred <- tree_pred

# Showing erro by visuals
tree_error <- tree_pred_test - data$y

# Visualize linear error
ggplot(data, aes(colour = train_data)) +
  geom_point(aes(x = tree_pred, y = tree_error)) +
  geom_smooth(aes(x = tree_pred, y = tree_error), method = 'loess', span = 0.75)

tree_mse_train <- mean((data$y[train_idx] - tree_pred[train_idx])^2)
tree_mse_test <- mean((data$y[test_idx] - tree_pred[test_idx])^2)

#data_ord <- data[order(data$x), ]
# Add plot over puntos
tree_puntos <- puntos +
  geom_line(
    data = data,
    aes(x = x, y = tree_pred),
    color = "blue",
    linewidth = 1
  )
tree_puntos

# R^2 = 1 - SSE/SST
tree_SSE <- sum((test$y - tree_pred_test)^2)
tree_SST <- sum((test$y - mean(test$y))^2)

tree_R2_test <- 1 - tree_SSE / tree_SST
tree_R2_test


tree_model$cptable
cp_min <- tree_model$cptable[which.min(tree_model$cptable[, 'xerror']), 'CP']

rpart.plot::prp(tree_model)


# 4. Adjusting nnet
set.seed(300)
# Trainin by grid that create all possible combinations
net_model <- train(
  y ~ x,
  tunedGrid = expand.grid(size = 1:15, decay = seq(0, 1, length.out = 15)),
  data = train,
  method = 'nnet',
  trControl = trainControl(method = 'cv', number = 10, verboseIter = FALSE),
  linout = TRUE,
  maxit = 1000,
  trace = FALSE
)

net_model
set.seed(300)
nn_pred <- predict(net_model, data)
nn_pred_test <- predict(net_model, test)
data$nn_pred <- nn_pred

nn_error <- nn_pred_test - data$y

# Visualize linear error
ggplot(data, aes(colour = train_data)) +
  geom_point(aes(x = nn_pred, y = nn_error)) +
  geom_smooth(aes(x = nn_pred, y = nn_error), method = 'loess')

nn_mse_train <- mean((data$y[train_idx] - nn_pred[train_idx])^2)
nn_mse_test <- mean((data$y[test_idx] - nn_pred[test_idx])^2)

# R^2 = 1 - SSE/SST
nn_SSE <- sum((test$y - nn_pred_test)^2)
nn_SST <- sum((test$y - mean(test$y))^2)

nn_R2_test <- 1 - nn_SSE / nn_SST
nn_R2_test


net_puntos <- puntos +
  geom_line(
    data = data,
    aes(x = x, y = nn_pred),
    color = "purple",
    linewidth = 1
  )
net_puntos
