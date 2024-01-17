## ML-algorithms
This repository provides my own implementation of machine learning library on native Python 3.12 + NumPy. Some architectural ideas, features and models' parameters
were taken from scikit-learn.

## Directories

### tree 
- DecisionTreeClassifier. Builds a decision tree by ID3 algorithm. Other algorithms will be added later.

### ensemble
- RandomForestClassifier. Builds n DecisionTreeClassifiers and returns their average predictions

### datasets
- Provides a few datasets downloaded from opened sources and simple preprocessing functions for each dataset

### linear_models

### metrics
- Various metrics and criteria for different tasks

### model_selection
- Classes for hyperparameters tuning 

### preprocessing 
- Diverse functions for data preprocessing, e.g. binning, splitting, scaling, etc.


