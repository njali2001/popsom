import popsom as som
import numpy as np
import pandas as pd
from sklearn import datasets	# Load iris dataset
iris = datasets.load_iris()
labels = iris.target
data = pd.DataFrame(iris.data[:, :4])
data.columns = iris.feature_names
m = som.map()  
m.fit(data,labels)
m.significance()
m.convergence()
m.projection()
m.neuron(3,4)
m.marginal(3)
m.starburst()
