[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **DEDA_SVM_Linear** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet : DEDA_SVM_Linear

Published in : 'DEDA Unit 7: Support Vector Machines (SVM)'

Description : 'Determines and plots the decision boundaries of a linear SVM classifier for different regularization parameters C.'

Keywords : Support vector machines, SVM, classification

See also : 'DEDA_SVM_Nonlinear, DEDA_SVM_Spiral, DEDA_SVM_Swiss'

Author : Georg Keilbar

Submitted : October 16 2018 by Georg Keilbar

```

![Picture1](linear_0.05.png)

![Picture2](linear_1.png)

### PYTHON Code
```python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#simulation of 40 data points
np.random.seed(1)
X = np.r_[np.random.randn(20, 2) - [1.5, 1.5], 
          np.random.randn(20, 2) + [1.5, 1.5]]
Y = [0] * 20 + [1] * 20


def plot_svm_linear(X, Y, penalty):
    #fit the model
    fignum = 1
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)

    #get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    #get the margin and the parallels
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    lower = yy - np.sqrt(1 + a ** 2) * margin
    upper = yy + np.sqrt(1 + a ** 2) * margin

    #plot the separating plane and the parallels
    plt.figure(fignum, figsize=(4, 4))
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, lower, 'k--')
    plt.plot(xx, upper, 'k--')

    #plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=5, cmap='bwr',
                edgecolors='k')
    #mark the support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=20, edgecolors='k')

    plt.axis('tight')
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    
    fignum = fignum + 1
    
plot_svm_linear(X, Y, 1000)
plt.savefig('linear_1.png', transparent=True, dpi=200)
plt.clf()

plot_svm_linear(X, Y, 0.05)
plt.savefig('linear_0.05.png', transparent=True, dpi=200)
plt.clf()
```

automatically created on 2018-10-18