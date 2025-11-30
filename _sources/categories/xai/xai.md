# Explainable AI (XAI)

Accurate predictions are all well and good, but it's hard to be confident about them if you can't figure out WHY the model came to that conclusion. Explainable AI (XAI) help reduce the "black box" factor of a model by helping to explain its decision-making process.

LIME and SHAP were mentioned as Desired Proficiencies in one of the postings I looked at.

## [SHapley Additive exPlanations (SHAP)](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)
- uses game theory to determine the contribution of each feature  to the prediction
- follow along with the example for more details (will need to make a notebook)
- see the [SHAP Notebook](./shapley_xai_intro.ipynb)

## [Local Interpretable Model-agnostic Explanations (LIME)](https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-explainable-aixai-using-lime/)
- treats a supervised ML model as a black box (hence model-agnostic)
- creates 5000 samples (by default). This means it makes 5000 feature vectors following a normal distribution
- it inferences the model to make predictions on each of the samples and obtains the target variable (predictions)
- it weighs each feature to determine how close each feature is to the original sample/observation (need to drill down into this more)
- uses feature selection technique like Lasso Regression or Ridge Regression to determine which features were most important to that prediction
- follow along with the example for more details (will need to make a notebook)

```{toctree}
:glob:

./*
```