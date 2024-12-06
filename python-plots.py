import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec

# Set style for prettier plots
sns.set_style('whitegrid')
sns.set_context('talk')

# Load the Diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Use one feature for easy visualization (e.g., BMI at index 2)
X = X[:, np.newaxis, 2]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Polynomial degrees to test
degrees = [1, 3, 5, 7, 9]
train_errors = []
test_errors = []

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

# Degrees to visualize in the top row
degrees_to_visualize = [1, 5, 9]

# Residual plots for these degrees in the second row
residual_degrees = [1, 9]

# Create figure and GridSpec
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig)

# Top row: three polynomial fits
axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]

# Define custom colors for the top row
top_train_color = "#2a9d8f"  # teal
top_line_color = "#e76f51"   # salmon

for i, d in enumerate(degrees_to_visualize):
    poly_viz = PolynomialFeatures(degree=d)
    X_train_poly_viz = poly_viz.fit_transform(X_train)
    model_viz = LinearRegression()
    model_viz.fit(X_train_poly_viz, y_train)
    
    # Create a smooth curve for the polynomial fit
    X_curve = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
    X_curve_poly = poly_viz.transform(X_curve)
    y_curve = model_viz.predict(X_curve_poly)
    
    # Plot the training data and the polynomial fit with new colors
    axes_top[i].scatter(X_train, y_train, color=top_train_color, alpha=0.7, label='Training Data')
    axes_top[i].plot(X_curve, y_curve, color=top_line_color, linewidth=3, label=f'Degree {d}')
    axes_top[i].set_title(f'Polynomial Degree {d}', fontsize=18, pad=10)
    axes_top[i].set_xlabel('Feature (BMI)', fontsize=14)
    axes_top[i].set_ylabel('Target', fontsize=14)
    axes_top[i].legend(fontsize=12)

# Bottom row: 2 residual plots + training vs test error
ax_res1 = fig.add_subplot(gs[1, 0])
ax_res2 = fig.add_subplot(gs[1, 1])
ax_err = fig.add_subplot(gs[1, 2])

for i, d in enumerate(residual_degrees):
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_test_pred = model.predict(X_test_poly)
    residuals = y_test - y_test_pred

    ax = ax_res1 if i == 0 else ax_res2
    ax.scatter(y_test_pred, residuals, color='purple', alpha=0.7)
    ax.hlines(y=0, xmin=y_test_pred.min(), xmax=y_test_pred.max(), colors='red', linestyles='dashed')
    ax.set_title(f'Residual Plot (Degree={d})', fontsize=16, pad=10)
    ax.set_xlabel('Predicted Values', fontsize=14)
    ax.set_ylabel('Residuals', fontsize=14)

# Keep the original palette for training vs test error plot
palette = sns.color_palette("deep", 3)

# Plot training vs test errors
ax_err.plot(degrees, train_errors, label='Training Error', marker='o', markersize=8, linewidth=3, color=palette[0])
ax_err.plot(degrees, test_errors, label='Test Error', marker='s', markersize=8, linewidth=3, color=palette[1])
ax_err.set_title('Training vs Test Errors', fontsize=18, pad=10)
ax_err.set_xlabel('Polynomial Degree', fontsize=14)
ax_err.set_ylabel('Mean Squared Error', fontsize=14)
ax_err.legend(fontsize=12)

plt.tight_layout()
plt.show()
