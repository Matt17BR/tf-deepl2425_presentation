# Load necessary libraries
library(MASS)       # For the Boston dataset
library(ggplot2)    # For plotting
library(reshape2)   # For data manipulation

# Set seed for reproducibility
set.seed(123)

# Load the Boston housing data
data(Boston)

# Prepare the data
X <- as.matrix(Boston[, -14])  # Exclude the target variable 'medv'
y <- Boston$medv

# Split into training and test sets
train_index <- sample(1:nrow(X), size = 0.7 * nrow(X))
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Define a range of model complexities (number of features)
max_features <- 200  # Max number of features to consider
n_real_features <- ncol(X_train)  # Number of real features

# Initialize vectors to store errors
train_errors <- numeric(max_features)
test_errors <- numeric(max_features)

# Loop over increasing number of features
for (p in 1:max_features) {
  # Determine the number of random features to add
  n_random_features <- max(0, p - n_real_features)
  
  # Generate random features for training and test sets
  if (n_random_features > 0) {
    random_train <- matrix(rnorm(nrow(X_train) * n_random_features), nrow = nrow(X_train))
    random_test <- matrix(rnorm(nrow(X_test) * n_random_features), nrow = nrow(X_test))
    
    # Combine real and random features
    X_train_p <- cbind(X_train, random_train)
    X_test_p <- cbind(X_test, random_test)
  } else {
    X_train_p <- X_train[, 1:p, drop = FALSE]
    X_test_p <- X_test[, 1:p, drop = FALSE]
  }
  
  # Prepare data frames for lm()
  df_train <- data.frame(y = y_train, X_train_p)
  df_test <- data.frame(y = y_test, X_test_p)
  
  # Build the formula for lm()
  predictors <- paste(colnames(df_train)[-1], collapse = " + ")
  formula <- as.formula(paste("y ~", predictors))
  
  # Fit linear model
  model <- lm(formula, data = df_train)
  
  # Predict on training and test sets
  y_pred_train <- predict(model, newdata = df_train)
  y_pred_test <- predict(model, newdata = df_test)
  
  # Compute mean squared errors
  train_errors[p] <- mean((y_train - y_pred_train)^2)
  test_errors[p] <- mean((y_test - y_pred_test)^2)
}

# Create a data frame for plotting
df_errors <- data.frame(
  Complexity = 1:max_features,
  Training_Error = train_errors,
  Test_Error = test_errors
)

# Melt the data for ggplot
df_melted <- melt(df_errors, id.vars = "Complexity", variable.name = "Type", value.name = "Error")

# Plot the errors vs model complexity
ggplot(df_melted, aes(x = Complexity, y = Error, color = Type)) +
  geom_line(size = 1) +
  xlab("Model Complexity (Number of Features)") +
  ylab("Mean Squared Error") +
  ggtitle("Overfitting Phenomenon in Linear Regression") +
  theme_minimal()
