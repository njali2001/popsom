# POPSOM ![CI status](https://img.shields.io/badge/build-passing-brightgreen.svg)

POPSOM is a Python library for dealing with population-base Self-Organizing Maps. This work was derived from [R-based POPSOM](https://github.com/lutzhamel/popsom) which developed and maintained by [Dr. Lutz Hamel](https://www.cs.uri.edu/about-us/people/lutz-hamel/) and his former students. 

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

### How to install
* Copy the popsom.py into python work dict.



## Example 1: Animal data

* We have 13 different kinds of animals with 13 different features.

|             | dove | hen  | duck | owl  |eagle | dog  | wolf | cat  |tiger | lion |horse | cow  |
| ----------- |:----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:| ----:|
| **Small**   |  1   |  1   |   1  |  1   |  0   |  0   |  0   |  1   |  0   |  0   |  0   |  0   | 
| **Medium**  |  0   |  0   |   0  |  0   |  1   |  1   |  1   |  0   |  0   |  0   |  0   |  0   |
| **Big**     |  0   |  0   |   0  |  0   |  0   |  0   |  0   |  0   |  1   |  1   |  1   |  1   |
| **2 legs**  |  1   |  1   |   1  |  1   |  1   |  0   |  0   |  0   |  0   |  0   |  0   |  0   | 
| **4 legs**  |  0   |  0   |   0  |  0   |  0   |  1   |  1   |  1   |  1   |  1   |  1   |  1   |
| **hair**    |  0   |  0   |   0  |  0   |  0   |  1   |  1   |  1   |  1   |  1   |  1   |  1   |
| **Hooves**  |  0   |  0   |   0  |  0   |  0   |  0   |  0   |  0   |  0   |  0   |  1   |  1   | 
| **Mane**    |  0   |  0   |   0  |  0   |  0   |  0   |  1   |  0   |  0   |  1   |  1   |  0   |
| **Feathers**|  1   |  1   |   1  |  1   |  1   |  0   |  0   |  0   |  0   |  0   |  0   |  0   |
| **Hunt**    |  0   |  0   |   0  |  1   |  1   |  0   |  1   |  1   |  1   |  1   |  0   |  0   | 
| **Run**     |  0   |  0   |   0  |  0   |  0   |  1   |  1   |  0   |  1   |  1   |  1   |  0   |
| **Fly**     |  1   |  0   |   1  |  1   |  0   |  0   |  0   |  0   |  0   |  0   |  0   |  0   |
| **Swim**    |  0   |  0   |   1  |  0   |  0   |  0   |  0   |  0   |  0   |  0   |  0   |  0   | 


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
```

* Initialize the model.
```python
m = som.map(xdim=10,ydim=5)
```

* Train the data.
```python
m.fit(attr,animal)
```

* Compute and display the starburst representation of clusters
```python
m.starburst()
```
<img src="image/animal_1.png"/>


## Example 2: Iris data

* Prepare the iris data for training.

```python
iris 	= datasets.load_iris()
labels 	= iris.target
data 	= pd.DataFrame(iris.data[:, :4])
data.columns = iris.feature_names
```

* Initialize the model.
```python
m = som.map(xdim=10,ydim=5,train=1000,norm=False) 
```

* Train the data.
```python
m.fit(data,labels)
```

* Compute the relative significance of each feature and plot it
```python
m.significance()
```
<img src="image/sign.png"/>

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
<img src="image/iris.png"/>

* Plot that shows the marginal probability distribution of the neurons and data
```python
m.marginal(0)
m.marginal(1)
m.marginal(2)
m.marginal(3)
```

<p float="left">
  <img src="image/Figure_1.png" width="400" />
  <img src="image/Figure_2.png" width="400" /> 
</p>

<p float="left">
  <img src="image/Figure_3.png" width="400" />
  <img src="image/Figure_4.png" width="400" /> 
</p>

* Print the association of labels with map elements
```python
m.projection()
```

* Returns the contents of a neuron at (x,y) on the map as a vector
```python
m.neuron(6,3)
```

## Reference Thesis
[Yuan, Li](mailto:li_yuan@my.uri.edu), "[Implementation of Self-Organizing Maps with Python](https://digitalcommons.uri.edu/theses/1244)" (2018).

## License
[MIT](https://choosealicense.com/licenses/mit/)
