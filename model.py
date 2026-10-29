
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Loading the dataset
data = pd.read_csv(
    r"house_price_train_data.csv"
)

x_train = data["size_sqft"].to_numpy(dtype=float)
y_train = data["price_1000s"].to_numpy(dtype=float)


# 2. Normalize the input features

x_mean = np.mean(x_train)
x_std = np.std(x_train)

x_norm = (x_train - x_mean) / x_std


# 3. Cost function

def compute_cost(x, y, w, b):
    m = len(y)

    y_pred = w * x + b
    error = y_pred - y

    cost = (1 / (2 * m)) * np.sum(error ** 2)

    return cost


# 4. Gradient descent


def gradient_descent(x, y, w, b, alpha, iterations):
    m = len(y)

    for i in range(iterations):

        # Make predictions
        y_pred = w * x + b

        # Calculate prediction errors
        error = y_pred - y

        # Calculate gradients
        dj_dw = (1 / m) * np.sum(error * x)
        dj_db = (1 / m) * np.sum(error)

        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Print cost every 1000 iterations
        if i % 1000 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost = {cost:.2f}")

    return w, b


# 5. Initialize parameters and train the model
w = 0.0
b = 0.0

alpha = 0.01
iterations = 10000

final_w, final_b = gradient_descent(
    x_norm, y_train, w, b, alpha, iterations
)


# 6. Predict the price of a new house

def predict_price(size_sqft):
    # Normalize the new house size using training data statistics
    x = (size_sqft - x_mean) / x_std

    # Predict the price
    price = final_w * x + final_b

    return price


# 7. Plot the training data and regression line

plt.scatter(x_train, y_train, label="Actual data")

x_line = np.linspace(min(x_train), max(x_train), 100)
y_line = [predict_price(size) for size in x_line]

plt.plot(x_line, y_line, color="red", label="Regression line")

plt.xlabel("House Size (sqft)")
plt.ylabel("Price (thousands)")
plt.title("House Price Prediction")
plt.legend()
plt.show()

if __name__ == "__main__":
    final_w, final_b = gradient_descent(
        x_norm, y_train, 0.0, 0.0, 0.01, 100000
    )

    print("Model training complete!")
