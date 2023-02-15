import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
import io
import requests

url = "https://gitlab.com/bivl2ab/academico/cursos-uis/ai/ai-2-uis-student/-/raw/master/data/leaf_class_1.csv"
s = requests.get(url).content
data = pd.read_csv(io.StringIO(s.decode('utf-8')),error_bad_lines=False, header=None, usecols=[0,4,5,7], names=['Specie', 'Elongation', 'Convexity', 'Circularity'] )
Geranium = data[['Elongation', 'Convexity',  'Circularity']][data.Specie==36].values
Populus_alba = data[['Elongation', 'Convexity', 'Circularity',]][data.Specie==15].values
X = np.vstack([Geranium, Populus_alba])
Y = np.hstack([np.zeros(Geranium.shape[0]), np.ones(Populus_alba.shape[0])])
#===============
# First subplot
#===============
# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=(15,5))
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1)
# plot a scatters points
ax.scatter(Populus_alba[:, 0], Populus_alba[:, 1], label='Populus alba = 1')
ax.scatter(Geranium[:, 0], Geranium[:, 1], label='Geranium = 0')

# Use newton method
clf1 = LogisticRegression(random_state=0, multi_class='ovr', solver='newton-cg').fit(X[:,:-1], Y)
y = (-(clf1.coef_[0][0]*X[:,0]) - clf1.intercept_[0]) /clf1.coef_[0][1]
ax.plot(X[:,0],y, c='r')
ax.set_title(r'Simple logistic regression: $Class = logit(\omega_{0} + \omega_{1}E + \omega_{2}c_1 + \epsilon )$')
ax.set_xlabel(r'Elongation ($E$)')
ax.set_ylabel(r'Circularity ($c_1$)')
plt.legend()
#===============
# Second subplot
#===============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
# plot a 3D wireframe like in the example mplot3d/wire3d_demo

clf2 = LogisticRegression(random_state=0, multi_class='ovr', solver='newton-cg').fit(X, Y)

def plane(grid):
  Z = (-(clf2.coef_[0][0]*grid[:,0]) -(clf2.coef_[0][1]*grid[:,1]) - clf2.intercept_[0]) /clf2.coef_[0][2]

  return Z 

# create a wiremesh for the plane that the predicted values will lie
xx, yy = np.meshgrid(X[:, 0], X[:, 1])
combinedArrays = np.vstack((xx.flatten(), yy.flatten())).T
Z = plane(combinedArrays)

ax.scatter(Populus_alba[:, 0], Populus_alba[:, 1], Populus_alba[:, 2], label='Populus alba = 1')
ax.scatter(Geranium[:, 0], Geranium[:, 1], Geranium[:, 2], label='Geranium = 0')
ax.set_title(r'Multiple logistic regression: $Class = logit(\omega_{0} + \omega_{1}E + \omega_{2}c_1 +  \omega_{2}c_2 + \epsilon )$')
ax.set_xlabel(r'Elongation ($E$)')
ax.set_ylabel(r'Circularity ($c_1$)')
ax.set_zlabel(r'Convexity ($c_2$)')
plt.legend(loc='center right')
ax.plot_trisurf(combinedArrays[:, 0], combinedArrays[:, 1], Z, alpha=0.5, color='r')
plt.show()