# Chapter 24 — Hyperparameter Tuning

## Overview

Hyperparameter tuning is the process of searching for suitable hyperparameter values that help a Machine Learning model perform well and generalize to unseen data.

This chapter focused on:

- Hyperparameters and their role in model training
- Parameters vs hyperparameters
- Grid Search
- Random Search
- Cross-Validation
- GridSearchCV
- Random Forest hyperparameters
- Baseline vs tuned model comparison

## Hyperparameters vs Parameters

### Model Parameters
Parameters are learned automatically from the training data.

Examples:
- Weights
- Coefficients
- Decision-tree split conditions

### Hyperparameters
Hyperparameters are selected before or around the training process and control model complexity or learning behavior.

Examples:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Easy rule:

**Parameters = model learns them**  
**Hyperparameters = we choose/search for them**

## Hyperparameter Tuning Methods

### Grid Search

Grid Search evaluates every combination of values specified in a parameter grid.

For this practical:

- `n_estimators`: 50, 100, 200
- `max_depth`: 5, 10, 15, None
- `min_samples_split`: 2, 5
- `min_samples_leaf`: 1, 2

This produced:

`3 × 4 × 2 × 2 = 48`

parameter combinations.

### Random Search

Random Search evaluates a selected number of randomly sampled combinations rather than every possible combination. It can be more efficient for larger search spaces.

## Cross-Validation

The practical used **5-fold cross-validation**.

The training data is divided into five folds. Each fold is used as validation data once while the remaining folds are used for training. The resulting scores are averaged.

This provides a more reliable estimate for comparing hyperparameter configurations.

## GridSearchCV

Scikit-Learn's `GridSearchCV` combines Grid Search with Cross-Validation.

The practical used:

```python
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
```

Important outputs:

- `best_params_` → best hyperparameter combination found during cross-validation
- `best_score_` → best mean cross-validation score
- `best_estimator_` → model trained using the selected configuration

## Practical Dataset

The **Breast Cancer dataset** from Scikit-Learn was used.

- 569 observations
- 30 input features
- Binary classification target
- 80% training data
- 20% test data
- `random_state=42`
- Stratified train/test split

## Baseline and Tuned Model

A baseline Random Forest classifier was trained before tuning.

The baseline test accuracy was:

**95.61%**

GridSearchCV then evaluated multiple Random Forest configurations using 5-fold cross-validation.

The best configuration found in the practical was:

- `max_depth = 10`
- `min_samples_leaf = 1`
- `min_samples_split = 2`
- `n_estimators = 200`

Best cross-validation accuracy:

**96.04%**

Tuned model test accuracy:

**95.61%**

Therefore, the test-set improvement was:

**0 percentage points**

## Key Learning

Hyperparameter tuning does not guarantee a higher test accuracy.

Its purpose is to systematically search for a suitable model configuration using a validation strategy. The final model must still be evaluated on unseen test data.

A tuned model should always be compared against a baseline rather than assuming that tuning automatically improves performance.

## Practical Workflow

`Dataset → Train/Test Split → Baseline Model → Hyperparameter Grid → GridSearchCV → Cross-Validation → Best Parameters → Tuned Model → Test Evaluation → Baseline vs Tuned Comparison`

## Tools Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Google Colab

## Key Takeaway

**Hyperparameter tuning is a model-selection process, not a guarantee of higher test accuracy.**
