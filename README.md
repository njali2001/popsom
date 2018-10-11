# POPSOM ![CI status](https://img.shields.io/badge/build-passing-brightgreen.svg)

POPSOM is a Python library for dealing with population-base self-organizing maps.

## Installation

### Requirements
* Python 3.3 and up
* numpy==1.13.1
* pandas==0.20.3
* seaborn==0.7.1
* scikit-learn==0.19.0
* statsmodels==0.8.0
* scipy==0.19.1
* matplotlib==2.0.2

`$ pip install -r requirements.txt`

## Example 1: Iris data

* Copy the popsom.py into python work dict.

* Load popsom.py .

```python
import popsom as som  
```

* Load pandas and sklearn for importing iris dataset.

```python
import pandas as pd
from   sklearn import datasets
```

* Prepare the iris data for training.

```python
iris 	= datasets.load_iris()
labels 	= iris.target
data 	= pd.DataFrame(iris.data[:, :4])
data.columns = iris.feature_names
m = som.map(xdim=10, ydim=5, train=1000)   # xdim 
```

* Initiate the model.

```python
m = som.map(xdim=10,ydim=5,train=1000,norm=False) 
# xdim,ydim - the dimensions of the map
# alpha - the learning rate, should be a positive non-zero real number
# train - number of training iterations
# norm - normalize the input data space
```
* Training the data.

```python
m.fit(data,labels)
```

* Compute the relative significance of each feature and plot it
```python
m.significance()
```
![significance](https://user-images.githubusercontent.com/8847441/46817774-b228e300-cd4d-11e8-9d24-c3be95be3395.png)

* Compute the convergence index of a map
```python
m.convergence()
```
* Evaluate the embedding of a map using the F-test and a Bayesian estimate of the variance in the training data
```python
m.embed()
```

* Measure the topographic accuracy of the map using sampling
```python
m.topo()
```

* Compute and display the starburst representation of clusters
```python
m.starburst()
```


## License
[MIT](https://choosealicense.com/licenses/mit/)
