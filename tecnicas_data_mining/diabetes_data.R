# data_prep.R
#===============================================================================
# Load and prepare Pima Indians Diabetes dataset
#===============================================================================

load_diabetes_data <- function(seed = 123, train_prop = 0.65) {
  # Required libraries
  library(mlbench)
  library(tidyverse)
  library(caret)

  # Import data
  data('PimaIndiansDiabetes')
  diabetes_data <- PimaIndiansDiabetes

  # Overview
  message(
    "Data loaded: ",
    nrow(diabetes_data),
    " rows, ",
    ncol(diabetes_data),
    " columns."
  )
  message("Target distribution:")
  print(prop.table(table(diabetes_data$diabetes)))

  # Predictors
  predictors <- diabetes_data |> dplyr::select(-diabetes)

  # Target for decision tree
  target_var <- "diabetes"

  # Train/test split
  set.seed(seed)
  train_idx <- createDataPartition(
    diabetes_data$diabetes,
    p = train_prop,
    list = FALSE
  )
  train_data <- diabetes_data[train_idx, ]
  test_data <- diabetes_data[-train_idx, ]

  # Return results as list
  list(
    full_data = diabetes_data,
    train = train_data,
    test = test_data,
    predictors = predictors,
    target = target_var
  )
}
