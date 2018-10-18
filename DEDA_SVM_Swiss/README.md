[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **DEDA_SVM_Swiss** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet : DEDA_SVM_Swiss

Published in : 'DEDA Unit 7: Support Vector Machines (SVM)'

Description : 'Determines and plots the decision boundary of a SVM classifier with polynomial kernel of order 2 using the Swiss banknote dataset.'

Keywords : Support vector machines, SVM, classification

See also : 'DEDA_SVM_Linear, DEDA_SVM_Nonlinear, DEDA_SVM_Spiral'

Author : Georg Keilbar

Submitted : October 16 2018 by Georg Keilbar

```

![Picture1](swiss.png)

### PYTHON Code
```python

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import svm

#Load data
swiss = genfromtxt('swiss.txt', delimiter='', skip_header=1)

Y = swiss[:,1]
X = swiss[:,[5,6]]

#Simple scatterplot
plt.scatter(X[:,0],X[:,1],c=Y,cmap='bwr',s=4)


def plot_svm_nonlinear(x, y, model_class, **model_params):
    #Fit model
    model = model_class(**model_params)
    model.fit(x, y)
    
    #Define grid
    h = .001     
    x_min, x_max = 7,13
    y_min, y_max = 7,13
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    #Prediction on grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    #Contour + scatter plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.4, cmap='coolwarm',s=4)
    plt.gca().set_aspect('equal', adjustable='box')

    return plt

plot_svm_nonlinear(X,Y,svm.SVC,C=10,kernel='poly',degree=2)
plt.savefig('swiss.png', transparent=True, dpi=200)
plt.clf()
```

automatically created on 2018-10-18