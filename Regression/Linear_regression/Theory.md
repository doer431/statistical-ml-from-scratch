# Simple Linear Regression: Theory and Metrics

## 1. The Model: Best Fit Line

Simple Linear Regression (SLR) is a statistical method used to model the linear relationship between a dependent variable ($\text{Y}$) and an independent variable ($\text{X}$).

The core equation for the predicted line ($\hat{Y}$) is:

$$Y = mX + c$$

Where:

$Y$: The dependent variable (the value being predicted).

$X$: The independent variable (the predictor).

$m$ (or slope): Represents the weightage of $X$, indicating how much $Y$ changes for a one-unit change in $X$.

$c$ (or $Y$-intercept): The value of $Y$ when $X$ is zero.

The goal of Linear Regression is to find the Best Fit Line—the line that passes closest to all the data points, minimizing the overall prediction error.

## 2. The Loss Function: Minimizing Error

A "perfect line" passing through every data point is often impossible. The Best Fit Line is determined by minimizing the total error between the actual observed values ($y_i$) and the values predicted by the model ($\hat{y}_i$).

### Sum of Squared Errors (SSE)

The most common method for calculating and minimizing this error is the Ordinary Least Squares (OLS) method, which uses the Sum of Squared Errors (SSE) as its Loss Function (or Cost Function).

The formula for SSE is:

$$\text{SSE} = \sum_{i=1}^{n} (e_{i})^2 = \sum_{i=1}^{n} (y_{i} - \hat{y}_{i})^2$$

Where $e_i = (y_i - \hat{y}_i)$ is the difference (residual) between the actual and predicted value.

### Why Use Squares?
We square the errors for two main reasons, as indicated in your notes:

To handle negative and positive residuals: Squaring ensures that errors from data points above the line ($y_i - \hat{y}_i > 0$) and below the line ($y_i - \hat{y}_i < 0$) do not cancel each other out.

Differentiability: The resulting function, $F(m, b) = \sum (y_{i} - m x_{i} - b)^2$, is continuous and differentiable everywhere. This is crucial because minimizing the error is achieved using calculus, specifically by taking the partial derivatives of the error function with respect to $m$ and $b$ and setting them to zero:


$$\frac{\partial F}{\partial m} = 0 \quad \text{and} \quad \frac{\partial F}{\partial b} = 0$$

The line found by this method is the one that minimizes the error function value $F(m, b)$.

## 3. Performance Metrics (Evaluation)

Once the model is trained (i.e., $m$ and $c$ are found), several metrics are used to evaluate its performance.

### A. Mean Squared Error ($\text{MSE}$)

MSE is the average of the squared errors. It's a key metric because it uses the same units as the loss function.

$$\text{MSE} = \frac{\text{SSE}}{n}$$

### B. Root Mean Squared Error ($\text{RMSE}$)

RMSE is the square root of the MSE. This metric is preferred for interpretability because it returns the error to the original units of the dependent variable ($Y$).

$$\text{RMSE} = \sqrt{\text{MSE}}$$

### C. Coefficient of Determination ($R^2$ Score)

The $R^2$ score (pronounced "R-squared") measures the proportion of the variance in the dependent variable ($Y$) that is predictable from the independent variable ($X$). It provides a good measure of model fit, ranging from $0$ to $1$.

$$\text{SST} = \sum_{i=1}^{n} (y_{i} - \overline{y})^2 \quad (\text{Total Sum of Squares})$$

$$R^2 = 1 - \frac{\text{SSE}}{\text{SST}}$$

Interpretation:

$R^2 = 1$: Model Perfect (The model explains 100% of the variance).

$R^2 = 0$: Model Useless (The model explains none of the variance, performing no better than simply predicting the average $\overline{y}$ for all points).

Example from notes: If $R^2 = 0.81$, it means $81\%$ of the variation in $Y$ is explained by the model (i.e., by the input $X$).