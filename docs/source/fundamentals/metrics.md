# Metrics
There are a lot of goodness of fit metrics and ways to characterize loss.

## Basics
- A model fits the data well when the differences between the **observed values** and **predicted values** are **small** and **unbiased**
- the differences between the `observed values` and `predicted values` are called `residuals`
- as the `goodness of fit` increases, the model is better fitted to the data

## Error
Before diving into ways to measure success, we should define the errors that we're tryinng to minimize.

### Total Prediction Error / Expected Generalization Error
- **Total Prediction Error**/**Expected Generalization Error** - error when applying a trained model to predict on unseen data. Can be broken down into 3 main components. 
- **Bias Error** - error from erroneous assumptions in the algorithm (leads to [underfitting](https://en.wikipedia.org/wiki/Overfitting#Underfitting)). 
- **Variance Error** - error from sensitivity to small fluctuations in training set (leads to [overfitting](https://en.wikipedia.org/wiki/Overfitting)). Fit the model too much and you'll end up fitting to the noise too.
- **Irreducible Error** - error inherent in any dataset or process that cannot be reduced regardless of model complexity or amount of data. This is the lower bound on total prediction error.

### Bias-Variance Tradeoff
- The **Bias-Variance Tradeoff** ([g4g](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)) is an inescapable tradeoff for supervised learning.
- you want your model to capture regularities in the training data and also generalize well to unseen data.
- **High-Bias** Models are simpler, can't capture as much detail, so they won't fit to noise, but they may miss patterns in the training data, resulting in underfitting.
- **High-Variance** Models can represent the training data set well but risk overfitting to noise or training data that isn't representative of the larger population.

(regression-metrics)=
## Regression Metrics
There are a lot of ways to score the accuracy of a regression model, each with advantages and disadvantages.

### Metrics Comparison
| Metric | Penalizes Large Errors | Sensitive to Outliers | Interpretability | Recommended |
| --- | --- | --- | --- | --- |
| MAE  | ✗ | - | +++ | ✓ |
| MSE  | ✓ | + | + | ✓ |
| RMSE | ✓ | + | ++ | ✓ |
| MAPE | ✗ | Varies | ++ | ✗ |

### Variable Definitions
  - $y$ = observed values
  - $y_i$ = observed value $i$
  - $\hat y_i$ = predicted value $i$
  - $\bar y_i$ = average of the observed values $i$
  - $n$ = sample size

(r-squared)=
### R2 (R-squared)
**R-squared** ([g4g](https://www.geeksforgeeks.org/machine-learning/python-coefficient-of-determination-r2-score/)) (also known as **Coefficient of Determination**) is the ratio of the variance that's explained by the model to the variance that's explained by a simple mean. 

It's usually $0-1$, though it *can* be negative if the model is *worse* at explaining variance than just guessing the mean regardless of inputs. 

$R^2$ always increases as more independent variables are added, whether or not those variables are useful predictors. Higher values indicate better fit ranging from `0` to `1`.

$$
\begin{align}
R^2  &= 1-\frac{RSS}{TSS} = 1-\frac{\sum_{i=1}^n(y_i-\hat y_i)^2}{\sum_{i=1}^n(y_i-\bar y_i)^2} \\
RSS &= \sum_{i=1}^n(y_i-\hat y_i)^2 = \text{Residual Sum of Squares} \\
TSS &= \sum_{i=1}^n(y_i-\bar y_i)^2 = \text{Total Sum of Squares}    \\
\end{align}
$$

<img src="../images/r_squared_visualization.png" alt="Pic" width="400" />

(adjusted-r-squared)=
### Adjusted R-Squared
- **Adjusted R-Squared** is a modification of **R-squared** that penalizes the inclusion of variables that don't actually contribute to prediction
- $\bar R^2=1-\Large\frac{RSS/DOF_{res}}{TSS/DOF_{tot}}$ $= 1-(1-R^2)\Large\frac{n-1}{n-p-1}$
- $DOF_{res}$ = Degrees of Freedom of population variance around mean
- $DOF_{tot}$ = Degrees of Freedom of population variance around model
- $p$ = total number of explanatory variables (inputs)

(mae)=
### MAE (Mean Absolute Error)
- **MAE** ses the ame scale as target variable, robust to outliers, difficult to take derivatives of due to absolute value
- $MAE = \frac{1}{n} \sum_{i=1}^n|y_i-\hat{y}_i|$

(nmae)=
### NMAE (Normalized Mean Absolute Error)
- **NMAE** normalizes the absolute error by the range of actual values, making it a relative relative metric
- $NMAE = \Large\frac{\frac{1}{n}\sum_{i=1}^n{|\hat{y}_i-y_i|}}{\frac{1}{n}\sum_{i=1}^n{|y_i|}} = \frac{MAE(y,\hat{y})}{mean(|y|)}$

(mse)=
### MSE (Mean Squared Error)
- **MSE** is vulnerable to outliers because the error is squared
- $MSE =\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2$

(huber-loss)=
### Huber Loss
- **Huber Loss** ([g4g](https://www.geeksforgeeks.org/machine-learning/sklearn-different-loss-functions-in-sgd/)) is a hybrid loss function that transitions from `MAE` to `MSE` for larger errors, providing balance between `MAE`'s robustness and `MSE`’s sensitivity to outliers.
- $L_\delta = \begin{cases}\frac{1}{2}(y-\hat{y})^2 &|y-\hat{y}|<\delta \\ \delta((y-\hat{y})-\frac{1}{2}\delta) & |y-\hat{y}|\ge\delta\end{cases}$

(rmse)=
### RMSE (Root Mean Squared Error)
- **RMSE** Same training results as using `MSE`, still vulnerable to outliers, compare with `MAE` to see prevalence of outliers
- $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2}$

(rmsle)=
### RMSLE (Root mean Squared Log Error)
- **RMSLE** uses logs make it relative metric (ignore scale of data), less vulnerable to outliers than $RMSE$, asymmetric (larger penalty if $\hat{y}_i < y_i$ than if $\hat{y}_i > y_i$)
- $RMSLE = \sqrt{\frac{1}{n}\sum_{i=1}^n (\ln{(\hat{y}_i+1)} - \ln{(y_i+1)})^2} =\Large\sqrt{\frac{1}{n}\sum_{i=1}^n (\ln{\frac{\hat{y}_i+1}{y_i+1}})^2}$
- <img src="../images/rmse_vs_rmsle.png" alt="RMSE vs RMSLE" width="350" />

(mape)=
### MAPE (Mean Absolute Percentage Error)
- AVOID **MAPE** LIKE THE PLAGUE. It fails if any $y_i=0$, higher penalty for small $y_i$, higher penalty for $\hat{y}_i > y_i$ than $\hat{y}_i < y_i$
- $MAPE = \Large\frac{1}{n}\sum_{i=1}^{n}{|\frac{y_i-\hat{y}_i}{y_i}|}*100$

(smape)=
### SMAPE (Symmetric Mean Absolute Percentage Error)
- AVOID **SMAPE**. It improves *somewhat* on `MAPE` but still controversial, not symmetric, and the equation itself varies by source
- $SMAPE = \Large\frac{1}{n}\sum_{i=1}^{n}{\frac{|y_i-\hat{y}_i|}{y_i+\hat{y}_i}}$

(bic)=
### BIC (Bayesian Information Criterion)
- **BIC** ([g4g](https://www.geeksforgeeks.org/machine-learning/bayesian-information-criterion-bic/)) evaluates goodness of fit $-2\ln(L)$ while penalizing complexity to avoid overfitting $k\ln(n)$
- $BIC = -2\ln(L) + k\ln(n)$
- $L$ = likelihood of the model given the data
- $k$ = number of parameters in the model
  
(aic)=
### AIC (Akaike Information Criterion)
- **AIC** ([wiki](https://en.wikipedia.org/wiki/Akaike_information_criterion)) estimates prediction error by estimating the amount of information lost by a model, dealing with the tradeoff between goodness of fit and model simplicity, i.e. it deals with the risks of both overfitting and underfitting.
- $AIC = 2k - 2 \ln(\hat L)$
- $\hat L$ = maximized value of the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function) for the model

### Regression Metric Equations
- $y$ = observed values
- $y_i$ = observed value $i$
- $\hat y_i$ = predicted value $i$
- $\bar y_i$ = average of the observed values $i$
- $n$ = sample size

$$
\begin{align}
R^2 &= 1-\frac{RSS}{TSS}=1-\frac{\sum_{i=1}^n(y_i-\hat y_i)^2}{\sum_{i=1}^n(y_i-\bar y_i)^2} \\
\bar R^2 &= 1-\frac{RSS/DOF_{res}}{TSS/DOF_{tot}} = 1-(1-R^2)\frac{n-1}{n-p-1} \\
MAE &= \frac{1}{n} \sum_{i=1}^n|y_i-\hat{y}_i| \\
NMAE &= \frac{\frac{1}{n}\sum_{i=1}^n{|\hat{y}_i-y_i|}}{\frac{1}{n}\sum_{i=1}^n{|y_i|}} = \frac{MAE(y,\hat{y})}{mean(|y|)} \\
MSE &=\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2 \\
\text{Huber Loss } L_\delta &= \begin{cases}\frac{1}{2}(y-\hat{y})^2 &|y-\hat{y}|<\delta \\ \delta((y-\hat{y})-\frac{1}{2}\delta) & |y-\hat{y}|\ge\delta\end{cases} \\
RMSE &= \sqrt{\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2} \\
RMSLE &= \sqrt{\frac{1}{n}\sum_{i=1}^n (\ln{(\hat{y}_i+1)} - \ln{(y_i+1)})^2} =\sqrt{\frac{1}{n}\sum_{i=1}^n (\ln{\frac{\hat{y}_i+1}{y_i+1}})^2} \\
MAPE &= \frac{1}{n}\sum_{i=1}^{n}{|\frac{y_i-\hat{y}_i}{y_i}|}*100 \\
SMAPE &= \frac{1}{n}\sum_{i=1}^{n}{\frac{|y_i-\hat{y}_i|}{y_i+\hat{y}_i}} \\
BIC &= -2\ln(L) + k\ln(n) \\
AIC &= 2k - 2 \ln(\hat L) \\
\end{align}
$$

(classification-metrics)=
## Classification Metrics
Accuracy for Classifiers is a little different, and sometimes your priorities change.
- Sometimes it's more important to catch **everything**, even if you catch some false positives
- Sometimes it's more important that you **only** catch stuff you're sure about, and **no** false positives

### Classification Results
- **TP (True Positive)**:  The model `correctly`   predicted a `positive` outcome
- **TN (True Negative)**:  The model `correctly`   predicted a `negative` outcome
- **FP (False Positive)**: The model `incorrectly` predicted a `positive` outcome (Type I error)
- **FN (False Negative)**: The model `incorrectly` predicted a `negative` outcome (Type II error)

(confusion-matrix)=
### Confusion Matrix
Can display classification results with a [Confusion Matrix](https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/) 
- the link also explains the rest of these details and has example `sklearn` code to make **Confusion Matrices**
- this one scores a model that was trying to label an image as containing a cat or an ant
it did kind of OK (labeled one cat as an ant, but got the others correct)
- ![ant-cat-confusion-matrix](../images/confusion_matrix.png)
- here's a mockup of one for a model trying to classify emails as real or spam (though without the counts)

| | Predicted as Spam | Predicted as Real |
| --- | --- | --- |
| Actually was Spam | **TP** (True Positive)  | **FN** (False Negative) |
| Actually was Real | **FP** (False Positive) | **TN** (True Negative)  |

### Classification Metric Equations
Can combine the true/false positives/negatives from the Confusion Matrix in multiple ways to get numerical scores.

$$
\begin{align}
\text{Accuracy} &= \frac{TP + TN}{TP + TN + FP + FN} \\
\text{Precision} &= \frac{TP}{TP + FP} \\
\text{Recall} &= \frac{TP}{TP + FN} \\
\text{F1 Score} &= 2 \frac{\text{Precision} * \text{Recall}}{\text{Precision} + \text{Recall}} \\
\text{Specificity} &= \frac{TN}{TN + FP} \\
\text{Type 1 Error} &= \frac{FP}{FP + TN} \\
\text{Type 2 Error} &= \frac{FN}{TP + FN} \\
\end{align}
$$

(accuracy)=
### Accuracy
**Accuracy** is how many predictions were correct out of the total predictions (can be misleading if one result is more dominant than another)
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

(precision)=
### Precision
**Precision** is how many **Positive Instances** you caught. A high **Precision** indicates that you're making few **False Positive** predictions. (important if you can't afford **False Positives**)
$$\text{Precision} = \frac{TP}{TP + FP}$$

(recall)=
### Recall (Sensitivity)
**Recall** (or **Sensitivity**) is proportion of **True Positives** detected out of all **Positive Instances**. High **Recall** indicates that you're capturing most of the **Positives**. (important if you can't afford **False Negatives**)
$$\text{Recall} = \frac{TP}{TP + FN}$$

(f1)=
### F1-Score
**F1-Score** is combines **Precision** and **Recall**, assuming **False Positives** and **False Negatives** are equally important. Useful for imbalanced datasets where **Precision** and **Recall** might be skewed.
$$\text{F1 Score} = 2 \frac{\text{Precision} * \text{Recall}}{\text{Precision} + \text{Recall}}$$

(specificity)=
### Specificity
**Specificity** is how good the model is at correctly identifying **Negative Instances**
$$\text{Specificity} = \frac{TN}{TN + FP}$$

(type-1-error)=
### Type 1 Error
**Type 1 Error** is error from **False Positives**
$$\text{Type 1 Error} = \frac{FP}{FP + TN}$$

(type-2-error)=
### Type 2 Error
**Type 2 Error** is error from **False Negatives**
$$\text{Type 2 Error} = \frac{FN}{TP + FN}$$

(support)=
### Support
**Support** is the number of actual occurrences of the class in the dataset. Imbalanced **support** in the training data may indicate structural weakness in the reported scores of the model. It's not a function of the model but the dataset. 

For instance, if you're classifying animals into `cats`, `dogs`, and `parrots`, the **support** for the parrot class in that dataset will be the number of `parrot` samples in the dataset. The **support** for the dataset as a whole will be the combined number of `cat`, `dog`, and `parrot` samples in the whole dataset. If you train your classifier but you don't have many `parrot` samples in the dataset (i.e. your training dataset has low `parrot` **support**), don't expect your model to be great at classifying `parrots` even if it ends up with high scores in the other metrics.