# Multiple Linear Regression (MLR)

## 1. The Model: Scalar and Matrix Formulation

Multiple Linear Regression extends the simple linear model by using two or more independent variables ($x_1, x_2, \dots, x_m$) to predict a dependent variable ($Y$).

### 1.1. Scalar Form (Prediction Equation)

The prediction for an output $\hat{Y}$ is a linear combination of the features:

$$\hat{Y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_m x_m$$

Where:

$\hat{Y}$: The predicted value.

$\beta_0$: The intercept.

$\beta_1, \dots, \beta_m$: The coefficients (weights) for each feature.

### 1.2. Matrix Formulation

To efficiently handle $n$ observations and $m$ features, the model is compactly expressed using matrices and vectors:

$$\hat{\mathbf{Y}} = \mathbf{X}\boldsymbol{\beta}$$

Component

Description

Dimension ($n$ observations, $m$ features)

$\hat{\mathbf{Y}}$

Predicted output vector ($n$ predictions)

$n \times 1$

$\mathbf{X}$

Design Matrix (includes a column of ones for $\beta_0$)

$n \times (m+1)$

$\boldsymbol{\beta}$

Coefficient Vector ($\beta_0, \beta_1, \dots, \beta_m$ )

$(m+1) \times 1$

## 2. The Loss Function: Ordinary Least Squares (OLS)

The goal is to find the coefficient vector $\boldsymbol{\beta}$ that minimizes the error between the actual values ($\mathbf{Y}$) and the predicted values ($\hat{\mathbf{Y}}$). This is achieved using the Sum of Squared Errors (SSE) as the loss function, which we will denote as $E$.

### 2.1. Sum of Squared Errors (SSE) in Matrix Form

The SSE ($E$) is calculated as the dot product of the error vector ($\mathbf{e} = \mathbf{Y} - \hat{\mathbf{Y}}$) with itself:

$$\text{SSE} = E = \mathbf{e}^T\mathbf{e} = (\mathbf{Y} - \hat{\mathbf{Y}})^T (\mathbf{Y} - \hat{\mathbf{Y}})$$

Substituting the model $\hat{\mathbf{Y}} = \mathbf{X}\boldsymbol{\beta}$:

$$E(\boldsymbol{\beta}) = (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$$

## 3. Finding the Coefficients: The Normal Equation

To find the optimal coefficient vector $\boldsymbol{\beta}$ that minimizes the error function $E(\boldsymbol{\beta})$, we use calculus, setting the partial derivative of the loss function with respect to $\boldsymbol{\beta}$ to zero:

$$\frac{\partial E}{\partial \boldsymbol{\beta}} = 0$$

Solving this results in the Normal Equation, which is the closed-form analytical solution for the OLS coefficients:

$$\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{Y}$$

## 4. Computational Considerations and Alternatives

### 4.1. Complexity of the Normal Equation

While the Normal Equation provides the exact solution, its calculation requires finding the inverse of the $(\mathbf{X}^T\mathbf{X})$ matrix.

This inverse calculation has a time complexity of approximately $O((m+1)^3)$, where $(m+1)$ is the number of coefficients (features plus the intercept).

For datasets with a very large number of features ($m$), this cubic complexity makes the computation time-consuming and inefficient.

### 4.2. Gradient Descent

Due to the computational cost of the Normal Equation on large datasets, an alternative optimization technique called Gradient Descent is typically used in machine learning.

Gradient Descent is an approximation technique that iteratively adjusts the coefficients ($\boldsymbol{\beta}$) in the direction of the steepest descent of the loss function until a minimum error value is approached.

This method is generally more scalable for high-dimensional data than the closed-form Normal Equation.