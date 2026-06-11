# K-Nearest Neighbors (KNN)

## Core Idea

KNN is a **lazy learner** — it does zero computation during training. The entire training dataset is memorized as-is, and all the work happens at prediction time. When a new point arrives, KNN looks at the `k` closest training points and lets them vote on the answer.

This is why `fit()` in `MyKnn` just stores `X_train` and `y_train` — there is literally nothing else to train.

---

## How Prediction Works (Step by Step)

Given a new query point `x`:

1. Compute the distance from `x` to **every** training point.
2. Sort all training points by distance (ascending).
3. Pick the top `k` closest points — the neighbors.
4. **Classification** → return the majority class label among the `k` neighbors.
5. **Regression** → return the mean of the `k` neighbors' target values.

This is exactly what `_predict_one()` does in `MyKnn`:

```python
distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))  # Step 1
k_indices = np.argsort(distances)[:self.k]                   # Steps 2 & 3
k_nearest_labels = self.y_train[k_indices]                   # Step 4 setup
most_common_label = max(counts, key=counts.get)              # Step 4 vote
```

---

## Distance Metric

Your `MyKnn` uses **Euclidean distance** (L2 norm):

$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

Other common metrics:

| Metric | Formula | Use Case |
|---|---|---|
| Euclidean (L2) | $\sqrt{\sum(p_i - q_i)^2}$ | Default, continuous features |
| Manhattan (L1) | $\sum\|p_i - q_i\|$ | More robust to outliers |
| Minkowski | $(\sum\|p_i - q_i\|^p)^{1/p}$ | Generalization of both |
| Hamming | Count of differing positions | Categorical features |

sklearn's `KNeighborsClassifier` uses Minkowski with `p=2` (Euclidean) by default.

---

## Choosing K

`k` is the single most important hyperparameter. Your sklearn notebook demonstrates both methods:

### Method 1 — √n Rule

```python
k = (lambda x: x+1 if x % 2 == 0 else x)(math.floor(np.sqrt(X_train.shape[0])))
```

Take the square root of the number of training samples, round down, then force it to be odd (to avoid ties in binary classification). Quick heuristic — not always optimal.

For 120 training samples: `√120 ≈ 10.95 → 10 → 11` (bumped to odd).

### Method 2 — Trial and Error (Accuracy Plot)

```python
for i in range(1, 120):
    knn = KNeighborsClassifier(n_neighbors=i)
    ...
    accuracy.append(accuracy_score(...))
plt.plot(range(1, 120), accuracy)
```

Iterate over all candidate `k` values, score each one, plot and pick the best. The error curve plot (training error vs test error) reveals the bias-variance tradeoff visually.

### The Bias-Variance Tradeoff with K

| K | Bias | Variance | Behavior |
|---|---|---|---|
| Small (k=1) | Low | High | Overfits — memorizes noise |
| Large (k=n) | High | Low | Underfits — predicts majority class for everything |
| Optimal | Balanced | Balanced | Generalizes well |

This is why the error plot in your notebook shows training error rising and test error falling as `k` increases — the sweet spot is somewhere in the middle.

> **Rule of thumb:** Always use odd `k` for binary classification to avoid ties.

---

## Applications

- **Recommendation Systems** — find users/items most similar to a target
- **Document Retrieval** — retrieve closest documents by embedding distance
- **Gene Expression Analysis** — classify tissue samples by expression profiles
- **Anomaly Detection** — points with no close neighbors are outliers
- **Image Recognition** — compare pixel/feature vectors (MNIST baseline)

---

## Limitations

### 1. Slow on Large Datasets
KNN stores the full training set. Every prediction requires computing distances to all `n` training points — **O(n × d)** per query. For large `n` or high `d`, this becomes very slow. sklearn mitigates this with `algorithm='kd_tree'` or `'ball_tree'`.

### 2. Curse of Dimensionality
In high-dimensional spaces, all points become approximately equidistant. The notion of "nearest neighbor" loses meaning because the ratio of max to min distance approaches 1. Fix: apply PCA or feature selection before KNN.

### 3. Feature Scaling is Mandatory
KNN is pure distance — if one feature has range `[0, 10000]` and another `[0, 1]`, the large-scale feature dominates all distance calculations completely. Always scale before using KNN.

### 4. High Memory Usage
The entire training set must be kept in memory at inference time — there is no compression into a model.

### 5. Sensitive to Irrelevant Features
A noisy or irrelevant feature adds noise to every distance calculation and can flip predictions. Feature selection matters.

### 6. Sensitive to Imbalanced Data
If one class has far more training points, it will statistically dominate majority voting even when the true nearest neighbors belong to a minority class.

---

## sklearn Key Parameters

```python
KNeighborsClassifier(
    n_neighbors=5,        # k value
    weights='uniform',    # 'uniform' or 'distance' (weight by 1/distance)
    metric='minkowski',   # distance metric
    p=2,                  # p=2 → Euclidean, p=1 → Manhattan
    algorithm='auto'      # 'kd_tree' or 'ball_tree' for large datasets
)
```