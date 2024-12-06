import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the Diabetes dataset
X, y = load_diabetes(return_X_y=True)

# For visualization purposes, use only one feature (e.g., BMI, which is feature index 2)
X = X[:, np.newaxis, 2]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

degrees = [1, 3, 5, 7, 9]  # Increasing polynomial degrees
train_errors = []
test_errors = []

# Fit models of varying complexity and record errors
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)

# Visualization: one example polynomial fit (e.g., with a high degree) on training data
degree_to_visualize = 9
poly_viz = PolynomialFeatures(degree=degree_to_visualize)
X_train_poly_viz = poly_viz.fit_transform(X_train)
model_viz = LinearRegression()
model_viz.fit(X_train_poly_viz, y_train)

# Create a smooth curve for plotting the polynomial function
X_curve = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
X_curve_poly = poly_viz.transform(X_curve)
y_curve = model_viz.predict(X_curve_poly)

# Plot the actual training points and the high-degree polynomial fit
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Training Data')
plt.plot(X_curve, y_curve, color='red', linewidth=2, label=f'Poly Degree {degree_to_visualize}')
plt.title(f'Training Data and Polynomial Fit (Degree={degree_to_visualize})')
plt.xlabel('Feature Value (BMI)')
plt.ylabel('Target')
plt.legend()

# Plot training vs test error for different polynomial degrees
plt.subplot(1, 2, 2)
plt.plot(degrees, train_errors, label='Training Error', marker='o')
plt.plot(degrees, test_errors, label='Test Error', marker='o')
plt.title('Overfitting in Action')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.legend()

plt.tight_layout()
plt.show()
