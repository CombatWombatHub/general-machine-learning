# Machine Learning
**Machine Learning (ML)** ([g4g](https://www.geeksforgeeks.org/machine-learning/machine-learning/)) is a branch of Artificial Intelligence that focuses on *models* and **algorithms** ([g4g](https://www.geeksforgeeks.org/machine-learning/machine-learning-algorithms/)) that let computers learn from data and improve from previous experience without being explicitly programmed. There are many **types of machine learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/types-of-machine-learning/)).

:::{note}
Some methods fit into multiple categories or can be adapted to be used for other categories. For the sake of brevity, these cases are not always mentioned here.
:::

:::{note}
I'm not sure how Neural Network's, Deep Learning, AutoEncoders, DenseNets, etc. fit into these categories. They may span multiple categories, or perhaps these are more traditional ML techniques.
:::

## Categories
- **Supervised Learning** - Use labeled data.
    - **Classification** - Predict *categorical* (discrete) values.
    - **Regression** - Predict continuous numerical values.
    - **Classification | Regression** - Some models can perform either Classification or Regression.
    - **Ensemble Learning** - Combine multiple models of either type into one better model.
        - **Bagging (Bootstrap Aggregating) Method** - Train models independently on different subsets of the data, then combine their predictions.
        - **Boosting Method** - Train models sequentially, each model focusing on errors of prior models, then do weighted combination of their predictions.
        - **Stacking (Stacked Generalization) Method** - train multiple different models (often different types), use predictions as inputs to final "meta-model".
- **Unsupervised Learning** - Use unlabeled data.
    - **Clustering** - Group data into clusters based on similarity.
        - **Centroid-Based (Partitioning) Clustering** cluster around centroids of points, choose number of clusters in advance.
        - **Distribution-Based Clustering** - cluster by mixture of probability distributions.
        - **Connectivity-Based (Hierarchical) Clustering** - cluster with tree-like nested groupings by connections between points.
        - **Density-Based (Model-Based) Clustering** - clusters as contiguous regions of high data density separated by areas of lower density.
    - **Dimensionality Reduction** - Simplify datasets by reducing features while keeping important information (often used to select features for other models).
    - **Association Rule Mining** - Discover rules where the presence of one item in a dataset indicates the probability of the presence of another.
- **Reinforcement Learning** - Agent learns by interacting with environment via trial and error and receiving reward feedback.
    - **Model-Based Methods** - interact with a simulated model of the environment, helping the agent plan actions by simulating potential results.
    - **Model-Free Methods** -  interact with the actual environment, learning directly from experience. 
- **Forecasting Models** - Use past data to predict future trends (often time series problems).
- **Semi-Supervised Learning** - Use some labeled data with more unlabeled data.
- **Self-Supervised Learning** - Generates its own labels from unlabeled data.

## Supervised Learning 
**Supervised Learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/)) is trained on *labeled* data

### Classification 
**Classification** ([g4g](https://www.geeksforgeeks.org/machine-learning/getting-started-with-classification/)) predicts *categorical* variables
- **KNN (K-Nearest Neighbors)** ([g4g](https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/)) - simple, looks at closest data points (neighbors) to make predictions based on similarity
- **Logistic Regression** ([g4g](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)) - Draws a sigmoid curve, predicts 0 or 1 if above or below curve. Despite "Regression" being in the name, it's for Classification
- **Single-Layer Perceptron** ([g4g](https://www.geeksforgeeks.org/python/single-layer-perceptron-in-tensorflow/)) - a single layer with a single neuron? Why?
- **SGD (Stochastic Gradient Descent) Classifier** ([g4g](https://www.geeksforgeeks.org/python/stochastic-gradient-descent-classifier/)) - adjust model parameters in the direction of the loss function's greatest gradient
- **Naive Bayes** ([g4g](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/)) (**Gaussian** ([g4g](https://www.geeksforgeeks.org/machine-learning/gaussian-naive-bayes/)), **Multinomial** ([g4g](https://www.geeksforgeeks.org/machine-learning/multinomial-naive-bayes/)), **Bernoulli** ([g4g](https://www.geeksforgeeks.org/machine-learning/bernoulli-naive-bayes/)), **Complement** ([g4g](https://www.geeksforgeeks.org/machine-learning/complement-naive-bayes-cnb-algorithm/))) - predicts the category of a data point with probability

### Regression
**Regression** ([g4g](https://www.geeksforgeeks.org/machine-learning/regression-in-machine-learning/)) predicts *continuous* variables
- **Linear Regression** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/)) - fit a straight line to the data with **Least Squares Method** ([g4g](https://www.geeksforgeeks.org/maths/least-square-method/))
- **Multiple Linear Regression** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-multiple-linear-regression-using-python/)) - Extends Linear Regression to use multiple input variables
- **Polynomial Regression** ([g4g](https://www.geeksforgeeks.org/machine-learning/python-implementation-of-polynomial-regression/)) - a polynomial curve fit.
- **Lasso Regression (L1 Regularization)** ([g4g](https://www.geeksforgeeks.org/machine-learning/ridge-regression-vs-lasso-regression/)) - regularized linear regression that avoids overfitting by penalizing the *absolute value* of large coefficients
- **Ridge Regression (L2 Regularization)** ([g4g](https://www.geeksforgeeks.org/machine-learning/ridge-regression-vs-lasso-regression/)) - regularized linear regression that avoids overfitting by penalizing the *square* of large coefficients

(ai-categories-ml-classification-regression)=
### Classification | Regression
Some types models can be used for either Classification or Regression
- **SVM (Support Vector Machine)** ([g4g](https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/)) | **SVR (Support Vector Regression)** ([g4g](https://www.geeksforgeeks.org/machine-learning/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/)) - use for Classification by finding a hyperplane that separates classes of data (SVM), or use for regression by finding the hyperplane that minimizes the residual sum of squares (SVR). Can be Linear or Non-Linear depending on the **Kernel** ([g4g](https://www.geeksforgeeks.org/machine-learning/linear-vs-non-linear-classification-analyzing-differences-using-the-kernel-trick/)) you select.
- **Multi-Layer Perceptron** ([g4g](https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/)) - classic neural network.
- **Decision Trees** ([g4g](https://www.geeksforgeeks.org/machine-learning/decision-tree-algorithms/)) (**Introduction** ([g4g](https://www.geeksforgeeks.org/machine-learning/decision-tree-introduction-example/))) (**Classification** ([g4g](https://www.geeksforgeeks.org/machine-learning/building-and-implementing-decision-tree-classifiers-with-scikit-learn-a-comprehensive-guide/))) (**Regression** ([g4g](https://www.geeksforgeeks.org/machine-learning/python-decision-tree-regression-using-sklearn/))) - hierarchical tree structure that works like a flow chart. splits data into branches based on feature values. Often used as building blocks for Ensemble methods. (**CART (Classification and Regression Trees)** ([g4g](https://www.geeksforgeeks.org/machine-learning/cart-classification-and-regression-tree-in-machine-learning/))) is based on (**ID3 (Iterative Dichotomiser 3)** ([g4g](https://www.geeksforgeeks.org/machine-learning/iterative-dichotomiser-3-id3-algorithm-from-scratch/))), and is a specific algorithm for building decision trees that can be used for both classification and regression.

### Ensemble Learning 
**Ensemble Learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/)) combines multiple models to make predictions

#### Bagging
**Bagging (Bootstrap Aggregating)** ([g4g](https://www.geeksforgeeks.org/machine-learning/What-is-Bagging-classifier/)) is an Ensemble Learning technique where multiple models are trained on different subsets of the data, then make predictions independently which are combined into a final result.
- [Random Forest](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/) ([Classification](https://www.geeksforgeeks.org/dsa/random-forest-classifier-using-scikit-learn/)) ([Regression](https://www.geeksforgeeks.org/machine-learning/random-forest-regression-in-python/)) ([Hyperparameter Tuning](https://www.geeksforgeeks.org/machine-learning/random-forest-hyperparameter-tuning-in-python/)) - create many decision trees, train each on random parts of data, combine results via voting (for classification) or averaging (for regression)
- **Random Subspace Method** ([mlm](https://www.machinelearningmastery.com/random-subspace-ensemble-with-python/)) - train on random subsets of input features to enhance diversity and improve generalization while reducing overfitting.

#### Boosting
**Boosting** ([g4g](https://www.geeksforgeeks.org/machine-learning/boosting-in-machine-learning-boosting-and-adaboost/)) is an Ensemble Learning techniue where you train models sequentially, each model focusing on errors of prior models, then do weighted combination of their predictions.
- [AdaBoost (Adaptive Boosting)](https://www.geeksforgeeks.org/machine-learning/implementing-the-adaboost-algorithm-from-scratch/) - for challenging examples, assign weights to data points, combine weak classifiers with weighted voting
- [GBM (Gradient Boosting Machines)](https://www.geeksforgeeks.org/machine-learning/ml-gradient-boosting/) - sequentially build decision trees, each tree correcting errors of previous ones
- [XGBoost (Extreme Gradient Boosting)](https://www.geeksforgeeks.org/machine-learning/xgboost/) - optimizes like regularization and parallel processing for robustness and efficiency
- [CatBoost (Categorical Boosting)](https://www.geeksforgeeks.org/machine-learning/catboost-ml/) - handles categorical features natively without extensive preprocessing

#### Stacking
**Stacking** ([mlm](https://machinelearningmastery.com/implementing-stacking-scratch-python/)) Stacks methods discussed above like K-Nearest Neighbors, Perceptron and Logistic Regression. The provided link demonstrates using k-Nearest Neighbors and Perceptron sub-models with a Logistic Regression model for aggregation, applying the resulting stack to a sonar dataset to predict rocks from metal cylinders based on sonar chirps bounced off of their surfaces.

## Unsupervised Learning
**Unsupervised Learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/unsupervised-learning/)) is trained on *unlabeled* data

### Clustering
**Clustering** ([g4g](https://www.geeksforgeeks.org/machine-learning/clustering-in-machine-learning/)) groups data into clusters based on similarity

#### Centroid-Based
**Centroid-Based (Partitioning) Clustering** cluster around centroids of points, choose number of clusters in advance.
- **K-Means Clustering** ([g4g](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)) - groups data into K clusters based on how close the points are to each other. Iteratively assigns points to the nearest centroid, recalculating centroids after each addition. Can use the **Elbow Method** ([g4g](https://www.geeksforgeeks.org/machine-learning/elbow-method-for-optimal-value-of-k-in-kmeans/)) to choose a good value for K
- **KMeans++ Clustering** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-k-means-algorithm/)) - improves K-Means by choosing initial cluster centers intelligently instead of randomly
- **K-Medoids Clustering** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-k-medoids-clustering-with-example/)) - similar to K-means, but uses actual data points (medoids) as the centers, making more robust to outliers
- **FCM (Fuzzy C-Means Clustering)** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-fuzzy-clustering/)) - similar to K-means but uses Fuzzy Clustering, allowing each data point to belong to multiple clusters with varying degrees of membership
- **K-Mode Clustering** ([g4g](https://www.geeksforgeeks.org/machine-learning/k-mode-clustering-in-python/)) - works on categorical data, unlike K-Means which is for numerical data

#### Distribution-Based
**Distribution-Based Clustering** clusters by mixture of probability distributions.
- **GMM (Gaussian Mixture Models)** ([g4g](https://www.geeksforgeeks.org/machine-learning/gaussian-mixture-model/)) - fits data as a weighted mixture of Gaussian distributions and assigns data points based on likelihood
- **DPMMs (Dirichlet Process Mixture Models)** ([g4g](https://www.geeksforgeeks.org/machine-learning/dirichlet-process-mixture-models-dpmms/)) - extension of **Gaussian Mixture Models** that can automatically decide the number of clusters based on the data
- **EM (Expectation-Maximization) Algorithm** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-expectation-maximization-algorithm/)) - Estimate unknown parameters using `E-Step` (`Expectation Step`) (calculating expected values of missing/hidden variables) and `M-Step` (`Maximization Step`) (maximizing log-likelihood to see how well the model explains the data)

#### Connectivity-Based
**Connectivity-Based (Hierarchical) Clustering** clusters with tree-like nested groupings by connections between points.
- **Hierarchical Clustering** ([g4g](https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/)) - create clusters by building a tree step-by-step, merging or splitting groups
- **Agglomerative Clustering** ([g4g](https://www.geeksforgeeks.org/machine-learning/agglomerative-methods-in-machine-learning/)) - (Bottom-up) start with each point as a cluster and iteratively merge the closest ones
- **Divisive Clustering** ([g4g](https://www.geeksforgeeks.org/artificial-intelligence/divisive-clustering/)) - (Top-down) starts with one cluster and splits iteratively into smaller clusters
- **Spectral Clustering** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-spectral-clustering/)) - groups data by analyzing connections between points using graphs
- **AP (Affinity Propagation)** ([g4g](https://www.geeksforgeeks.org/machine-learning/affinity-propagation-in-ml-to-find-the-number-of-clusters/)) - identify data clusters by sending messages between data points, calculates optimal number of clusters automatically

#### Density-Based
**Density-Based (Model-Based) Clustering** creates clusters as contiguous regions of high data density separated by areas of lower density.
- **Mean-Shift Clustering** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-mean-shift-clustering/)) - discovers clusters by moving points towards crowded areas
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** ([g4g](https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/)) - Groups points with sufficient neighbors, labels sparse points as noise
- **OPTICS (Ordering Points To Identify the Clustering Structure)** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-optics-clustering-explanation/)) - extends `DBSCAN` to handle varying densities

### Dimensionality Reduction
**Dimensionality Reduction** ([g4g](https://www.geeksforgeeks.org/machine-learning/dimensionality-reduction/)) is often used to prepare data for other models by reducing the number of features while retaining important information
- **PCA (Principal Component Analysis)** ([g4g](https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/)) - Reduces dimensions by transforming data into uncorrelated principal components.
- **ICA (Independent Component Analysis)** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-independent-component-analysis/))
- **t-SNE (t-distributed Stochastic Neighbor Embedding)** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-t-distributed-stochastic-neighbor-embedding-t-sne-algorithm/))
- **NMF (Non-negative Matrix Factorization)** ([g4g](https://www.geeksforgeeks.org/machine-learning/non-negative-matrix-factorization/)) - Breaks data into non-negative parts to simplify representation.
- **Isomap** ([g4g](https://www.geeksforgeeks.org/machine-learning/isomap-a-non-linear-dimensionality-reduction-technique/)) - Captures global data structure by preserving distances along a manifold.
- **LLE (Locally Linear Embedding)** ([g4g](https://www.geeksforgeeks.org/machine-learning/swiss-roll-reduction-with-lle-in-scikit-learn/)) - Reduces dimensions while preserving the relationships between nearby points.
- **LDA (Linear Discriminant Analysis)** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-linear-discriminant-analysis/)) - Reduces dimensions while maximizing class separability for classification tasks.

### Association Rule Mining
**Association Rule Mining** ([g4g](https://www.geeksforgeeks.org/machine-learning/association-rule/)) helps discover rules where the presence of one item in a dataset indicates the probability of the presence of another.
- **Apriori Algorithm** ([g4g](https://www.geeksforgeeks.org/machine-learning/apriori-algorithm/)) (**Implementation** ([g4g](https://www.geeksforgeeks.org/machine-learning/implementing-apriori-algorithm-in-python/))) - Finds patterns by exploring frequent item combinations step-by-step.
- **FP-Growth (Frequent Pattern-Growth)** ([g4g](https://www.geeksforgeeks.org/machine-learning/frequent-pattern-growth-algorithm/)) - An Efficient Alternative to Apriori. It quickly identifies frequent patterns without generating candidate sets.
- **ECLAT (Equivalence Class Clustering and Bottom-Up Lattice Traversal)** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-eclat-algorithm/)) - Uses intersections of itemsets to efficiently find frequent patterns.
- **Efficient Tree-based Algorithms** ([g4g](https://www.geeksforgeeks.org/dsa/introduction-to-tree-data-structure/)) - Scales to handle large datasets by organizing data in tree structures.

## Reinforcement Learning
**Reinforcement Learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/)) trains by interacting with an environment and giving feedback.

### Model-Based
 **Model-Based Methods** interact with a simulated model of the environment, helping the agent plan actions by simulating potential results.
- **MDPs (Markov Decision Processes)** ([g4g](https://www.geeksforgeeks.org/machine-learning/markov-decision-process/)) - describe step-by-step decisions where the results of actions are uncertain.  Evaluates all possible moves?
- **Monte Carlo Tree Search** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-monte-carlo-tree-search-mcts/)) - designed to solve problems with huge decision spaces, like the board game Go with $10^{170}$ possible board states, by building a search tree iteratively/randomly instead of exploring all possible moves.

### Model-Free
**Model-Free Methods** interact with the actual environment, learning directly from experience. 
- **Q-Learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/)) - makes trial-and-error guesses, building and updating a `Q-table` which stores `Q-values` which estimate how good it is to take a specific action in a given state.
- **Deep Q-Learning** ([g4g](https://www.geeksforgeeks.org/deep-learning/deep-q-learning/)) - Regular **Q-Learning** is good for small problems, but struggles on complex ones (like images) since the `Q-table` gets huge and computationally expensive. **Deep Q-Learning** fixes this by using a neural network to estimate the `Q-values` instead of a `Q-table`
- **SARSA (State-Action-Reward-State-Action)** ([g4g](https://www.geeksforgeeks.org/machine-learning/sarsa-reinforcement-learning/)) - helps an agent to learn an optimal policy by exploring the environment, taking actions, receiving feedback, and updating behavior for long-term rewards.
- **REINFORCE Algorithm** ([g4g](https://www.geeksforgeeks.org/machine-learning/reinforce-algorithm/)) - instead of estimating how good each action is, just *tries* actions and adjusts the chances of those actions based on the total reward afterwards
- **Actor-Critic Algorithm** ([g4g](https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/)) - combines an Actor (which selects actions via a Policy Gradient) and Critic (which evaluates the Actor via a Value Function), both of which learn (like your Loss function is getting smarter alongside your model)
- **A3C (Asynchronous Advantage Actor-Critic)** ([g4g](https://www.geeksforgeeks.org/machine-learning/asynchronous-advantage-actor-critic-a3c-algorithm/)) - uses multiple agents which learn in parallel, each interacting with their own private environments, then contribute their updates to a shared global model.

## Forecasting Models
**Forecasting Models** ([g4g](https://www.geeksforgeeks.org/machine-learning/time-series-analysis-and-forecasting/)) use past data to predict future trends (often time series problems).
- **ARIMA (Auto-Regressive Integrated Moving Average)** ([g4g](https://www.geeksforgeeks.org/r-language/model-selection-for-arima/)) - Combines `Autoregression` (`AR`), `Differencing` (`I`) and `Moving Averages` (`MA`) to capture patterns to predict future values based on historical data. Not great with seasonal data..
- **SARIMA (Seasonal ARIMA)** ([g4g](https://www.geeksforgeeks.org/machine-learning/sarima-seasonal-autoregressive-integrated-moving-average/)) - extension of **ARIMA** designed for time series data with seasonal patterns.
- **Exponential Smoothing** ([g4g](https://www.geeksforgeeks.org/artificial-intelligence/exponential-smoothing-for-time-series-forecasting/)) - assumes future patterns will be similar to more recent past data, focuses on learning average demand level over time. Simple and accurate for short-term forecasts, not great for long term forecasts. Uses `Simple`, `Double`, or `Holt-Winters` `Exponential Smoothing`.
- **RNNs (Recurrent Neural Networks)** ([g4g](https://www.geeksforgeeks.org/machine-learning/introduction-to-recurrent-neural-network/)) (**Tensorflow Example** ([g4g](https://www.geeksforgeeks.org/machine-learning/time-series-forecasting-using-recurrent-neural-networks-rnn-in-tensorflow/))) - neural networks where information can be passed backwards as well as forwards. They have many uses beyond forecasting, such as text generation
    - **LSTM (Long Short-Term Memory)** ([g4g](https://www.geeksforgeeks.org/deep-learning/multivariate-time-series-forecasting-with-lstms-in-keras/)) - use a memory mechanism to overcome the vanishing gradient problem
    - **GRU (Gated Recurrent Unit)** ([g4g](https://www.geeksforgeeks.org/deep-learning/multivariate-time-series-forecasting-with-grus/)) - efficient LStM combining input/forget gates and streamlining output mechanism

## Semi-Supervised Learning
**Semi-Supervised Learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-semi-supervised-learning/)) Uses some *labeled* data with a larger amount of *unlabeled* data.
- **Self-Training** ([g4g](https://www.geeksforgeeks.org/machine-learning/self-training-in-semi-supervised-learning/)) - The model is first trained on labeled data. It then predicts labels for unlabeled data, adding high-confidence predictions to the labeled set iteratively to refine the model. Includes **Pseudo Labelling** ([g4g](https://www.geeksforgeeks.org/machine-learning/pseudo-labelling-semi-supervised-learning/))
- **Co-Training** ([g4g](https://www.geeksforgeeks.org/machine-learning/what-is-co-training/)) - Two or more models are trained on different feature subsets of the data (like one model looks at the body of an email, another looks at the subject and sender, etc). Each model labels unlabeled data for the other, enabling them to learn from complementary views.
- **Multi-View Training** ([g4g](https://jmlr.org/papers/v21/18-794.html)) - A variation of co-training where models train on different data representations (e.g., images and text) to predict the same output.
- **Graph-Based Models (Label Propagation)** ([g4g](https://www.geeksforgeeks.org/machine-learning/ml-semi-supervised-learning/)) - Data is represented as a graph with nodes (data points) and edges (similarities). Labels are propagated from labeled nodes to unlabeled ones based on graph connectivity.
- **GAN (Generative Adversarial Network)** ([g4g](https://www.geeksforgeeks.org/deep-learning/generative-adversarial-network-gan/)) (**PyTorch Example** ([g4g](https://www.geeksforgeeks.org/deep-learning/generative-adversarial-networks-gans-in-pytorch/))) - create new, realistic data by learning from existing examples (creates good synthetic data)
- **Few-Shot Learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/few-shot-learning-in-machine-learning/)) - a **meta-learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/meta-learning-in-machine-learning/)) process where you train the model to learn quickly from new and unseen data, so you don't have to train it with a bunch of data initially. So I guess it does some quick additional learning when you "inference" it later?

## Self-Supervised Learning
**Self-Supervised Learning** ([g4g](https://www.geeksforgeeks.org/machine-learning/self-supervised-learning-ssl/)) Generates its own labels from unlabeled data. I haven't found specific examples for this yet, and most of the links I *have* found are for research papers.

## Links

```{toctree}
:glob:

./notes/**
./classification/classification
```
