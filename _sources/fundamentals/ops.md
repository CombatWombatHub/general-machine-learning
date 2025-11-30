# DevOps and MLOps

## DevOps
**ML Development Pipeline** In addition to training, the process of developing a new ML Model to solve a problem has a lot of steps if you want to do it right.
See GeeksForGeeks [Machine Learning Tutorial](https://www.geeksforgeeks.org/machine-learning/machine-learning/) for links to how-tos on each step.

### Data Preprocessing
Getting and cleaning the data, figuring out which features to use, potentially combine them to reduce the number of features while retaining the important information. 
Includes scaling and transforming.

### Exploratory Data Analysis
This seems like it should have a lot of overlap with the preprocessing step since it's about looking at features individually or looking at relationships between multiple features. Seems related to feature engineering and feature extraction, but  

### Model Evaluation and Tuning
Look at how well the model performed, tune its parameters improve its performance.

### Deploying
Once a [Machine Learning](https://www.geeksforgeeks.org/machine-learning/machine-learning/) model is trained with satisfactory performance, it needs to be [deployed](https://www.geeksforgeeks.org/machine-learning/machine-learning-deployment/) somewhere to be useful. You might [deploy it with a Streamlit App](https://www.geeksforgeeks.org/machine-learning/deploy-a-machine-learning-model-using-streamlit-library/) or set up an API with [Flask](https://www.geeksforgeeks.org/machine-learning/deploy-machine-learning-model-using-flask/) or [FastAPI](https://www.geeksforgeeks.org/machine-learning/deploying-ml-models-as-api-using-fastapi/) so other applications can access it.

## MLOps
After DevOps has successfully built and trained a model, MLOps takes over. It's focused on automating and managing the entire machine learning lifecycle, including deployment, monitoring and retraining while in production.