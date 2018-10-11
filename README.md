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

* Copy the popsom.py into python work dict.


## Example 1: Animal data


* We have 13 different animals with 13 different features.

|         | dove | hen  |duck  |owl   |eagle |dog   |wolf  |cat   |tiger |lion  |horse |cow   |
| ------- |:----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:|
| Small   |  1   |  1   |   1  |  1   |  0   |  0   |  0   |  1   |  0   |  0   |  0   |  0   | 
| Medium  |  0   |  0   |   0  |  0   |  1   |  1   |  1   |  0   |  0   |  0   |  0   |  0   |
| Big     |  0   |  0   |   0  |  0   |  0   |  0   |  0   |  0   |  1   |  1   |  1   |  1   |
| 2 legs  |  1   |  1   |   1  |  1   |  1   |  0   |  0   |  0   |  0   |  0   |  0   |  0   | 
| 4 legs  |  0   |  0   |   0  |  0   |  0   |  1   |  1   |  1   |  1   |  1   |  1   |  1   |
| hair    |  0   |  0   |   0  |  0   |  0   |  1   |  1   |  1   |  1   |  1   |  1   |  1   |
| Hooves  |  0   |  0   |   0  |  0   |  0   |  0   |  0   |  0   |  0   |  0   |  1   |  1   | 
| Mane    |  0   |  0   |   0  |  0   |  0   |  0   |  1   |  0   |  0   |  1   |  1   |  0   |
| Feathers|  1   |  1   |   1  |  1   |  1   |  0   |  0   |  0   |  0   |  0   |  0   |  0   |
| Hunt    |  0   |  0   |   0  |  1   |  1   |  0   |  1   |  1   |  1   |  1   |  0   |  0   | 
| Run     |  0   |  0   |   0  |  0   |  0   |  1   |  1   |  0   |  1   |  1   |  1   |  0   |
| Fly     |  1   |  0   |   1  |  1   |  0   |  0   |  0   |  0   |  0   |  0   |  0   |  0   |
| Swim    |  0   |  0   |   1  |  0   |  0   |  0   |  0   |  0   |  0   |  0   |  0   |  0   | 


* Load popsom, pandas and sklearn libraries.
```python
import popsom as som  
import pandas as pd
from   sklearn import datasets
```

* Prepare the data for training.
```python
animal = ['dove','hen','duck','owl','eagle','fox','dog','wolf','cat','tiger','lion','horse','cow']
attribute = [[1,0,0,1,0,0,0,0,1,0,0,1,0],
             [1,0,0,1,0,0,0,0,1,0,0,0,0],
             [1,0,0,1,0,0,0,0,1,0,0,1,1],
             [1,0,0,1,0,0,0,0,1,1,0,1,0],
             [0,1,0,1,0,0,0,0,1,1,0,0,0],
             [0,1,0,1,0,0,0,0,1,1,0,0,0],
             [0,1,0,0,1,1,0,0,0,0,1,0,0],
             [0,1,0,0,1,1,0,1,0,1,1,0,0],
             [1,0,0,0,1,1,0,0,0,1,0,0,0],
             [0,0,1,0,1,1,0,0,0,1,1,0,0],
             [0,0,1,0,1,1,0,1,0,1,1,0,0],
             [0,0,1,0,1,1,1,1,0,0,1,0,0],
             [0,0,1,0,1,1,1,0,0,0,0,0,0]]

attr = pd.DataFrame(attribute)
attr.columns = ['small','medium','big','2 legs','4 legs','hair','hooves','mane','feathers','hunt','run','fly','swim']
m = som.map(xdim=10,ydim=5)
```

* Training the data.
```python
m.fit(attr,animal)
```

* Compute and display the starburst representation of clusters
```python
m.starburst()
```
![animal](https://user-images.githubusercontent.com/8847441/46828059-3daf6d80-cd68-11e8-86c8-9071400cafcb.png)


## Example 2: Iris data

* Prepare the iris data for training.

```python
iris 	= datasets.load_iris()
labels 	= iris.target
data 	= pd.DataFrame(iris.data[:, :4])
data.columns = iris.feature_names
```

* Initiate the model.

```python
m = som.map(xdim=10,ydim=5,train=1000,norm=False) 
```
> xdim,ydim - the dimensions of the map
> alpha - the learning rate, should be a positive non-zero real number
> train - number of training iterations
> norm - normalize the input data space


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
![starburst](https://user-images.githubusercontent.com/8847441/46819587-1fd70e00-cd52-11e8-9959-1698a2119b32.png)

* Plot that shows the marginal probability distribution of the neurons and data
```python
m.marginal(0)
m.marginal(1)
m.marginal(2)
m.marginal(3)
```
![figure_1](https://user-images.githubusercontent.com/8847441/46819688-5f9df580-cd52-11e8-85d1-5b650702f756.png)
![figure_2](https://user-images.githubusercontent.com/8847441/46819689-5f9df580-cd52-11e8-856a-bb6e6d36c088.png)
![figure_3](https://user-images.githubusercontent.com/8847441/46819686-5f9df580-cd52-11e8-90d2-4a433b7dc6ec.png)
![figure_4](https://user-images.githubusercontent.com/8847441/46819687-5f9df580-cd52-11e8-84f4-01f2e736795a.png)

* Print the association of labels with map elements
```python
m.projection()
```

* Returns the contents of a neuron at (x,y) on the map as a vector
```python
m.neuron(6,3)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
