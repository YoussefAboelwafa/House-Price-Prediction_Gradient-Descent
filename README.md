# Gradiant-Descent_Linear-Regression

*This is Gradiant Descent Algorithm to minimize the cost function for linear regression models*
![gradiant-descent](https://github.com/YoussefAboelwafa/House-Price-Prediction_Gradiant-Descent/assets/96186143/09434ff9-66fd-4308-b623-b6beddafe258)

<a name="toc_40291_2.1"></a>

## Gradient descent summary
![functions](https://github.com/YoussefAboelwafa/House-Price-Prediction_Gradiant-Descent/assets/96186143/d4cf29bf-135a-46cf-b265-93445163ffed)

## Implement Gradient Descent

You will implement gradient descent algorithm for one feature. You will need three functions.

- `compute_gradient`
- `compute_cost`
- `gradient_descent`, utilizing compute_gradient and compute_cost
  <br>

![gd](https://github.com/YoussefAboelwafa/House-Price-Prediction_Gradiant-Descent/assets/96186143/63d8ead3-d25d-47f5-8ffa-1b48d4ee7227)


## Code Implementation
```python
import math, copy
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate the cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost


# Function to calculate the derivatives
def compute_gradient(x, y, w, b):
    # Number of training examples
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(
    x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function
):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        J_history.append(cost_function(x, y, w, b))
        p_history.append([w, b])

        if b == 0 and w == 0:
            break

    return w, b, J_history, p_history  # return w and J,w history for graphing
```


```python
# Load our data set
x_train = np.array([1.0, 3.0, 2.0, 5.0, 7.0])  # features
y_train = np.array([300.0, 500.0, 350.0, 600.0, 700.0])  # target value

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 100000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    tmp_alpha,
    iterations,
    compute_cost,
    compute_gradient,
)
print(f"(w,b) found by gradient descent: ({w_final:.4f},{b_final:.4f})")
```

    (w,b) found by gradient descent: (68.1034,244.8276)



```python
plt.scatter(x_train, y_train, marker="x", color="r")
plt.plot(x_train, w_final * x_train + b_final, color="b")
plt.xlabel("Size (sqtf)")
plt.ylabel("Price ($)")
plt.title("House Price Prediction")
plt.grid(True)
plt.show()
```



![png](images/output_2_0.png)




```python
plt.scatter(x_train, y_train, marker="x", color="r")
plt.plot(x_train, w_final * x_train + b_final, color="b")
plt.xlabel("Size (sqtf)")
plt.ylabel("Price (1000$)")
plt.title("House Price Prediction")
plt.grid(True)

target_x = 6.0
target_y = w_final * target_x + b_final
plt.scatter(target_x, target_y, marker="o", color="g", s=100)

plt.axhline(y=target_y, color="g", linestyle="--")
plt.axvline(x=target_x, color="g", linestyle="--")

plt.show()


print(f"6000 sqft house prediction {w_final*6.0 + b_final:0.1f} Thousand dollars")
```



![png](images/output_3_0.png)



    6000 sqft house prediction 653.4 Thousand dollars
