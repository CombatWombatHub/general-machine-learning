# Support Vector Machines (SVM)

- this article on the algorithm is really good https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/
  - helps me understand how it's machine learning
  - there's a penalty function that gets used during training
  - $\text{objective function} = \Large\frac{1}{\text(margin)} + \normalsize\lambda\sum(\text{penalty})$
  - try and minimize it (increasing margin and decreasing the penalty)
- often use [Hinge Loss](https://www.geeksforgeeks.org/machine-learning/hinge-loss-relationship-with-support-vector-machines/)
  - $L(y,f(x))=max(0,1−y∗f(x))$
  - $y=$ the actual class (classes `-1` or `1`) - so the true value
  - $f(x)=$ the output of the classifier for the datapoint - so the predicted value
  - ${}^1_2||w||^2 + C\sum_{i=1}^{n}(max(0.1-y_i(w\cdot x_i+b)))$
    - ${}^1_2||w||^2=$ regularization term (half of the squared Euclidian norm (i.e. L2 norm) of the weight vector $w$) (helps prevent overfitting by penalizing large weights)
  - find a hyperplane that separates classes with the widest possible margin to improve generalization
  - must balance maximization of this margin (which helps generalization and prevents overfitting) and potentially misclassifying some points (if that happens a lot, you're underfitting)
  - the term $C$ controls that trade-off: higher $C$
