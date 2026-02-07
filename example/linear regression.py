from typing import Literal
import numpy as np
x = np.array([[1000], [1200], [1500], [1800], [2000], [2200], [2500]])
y = np.array([[150], [180], [210], [240], [270], [290], [330]])

x_ =x.mean()
y_ = y.mean()

slope = ((x - x_) * (y - y_)).sum() / ((x - x_) ** 2).sum()
intercept = y_ - slope * x_

print(f"Linear Regression Equation: y = {slope:.2f}x + {intercept:.2f}")

# Predicting the price for a house with 1600 square feet
predicted_price = slope * 1600 + intercept