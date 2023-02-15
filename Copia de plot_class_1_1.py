import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import requests

url = "https://gitlab.com/bivl2ab/academico/cursos-uis/ai/ai-2-uis-student/-/raw/master/data/PRSA_data_class_1.csv"
s = requests.get(url).content
data = pd.read_csv(io.StringIO(s.decode('utf-8')),error_bad_lines=False)
X = data[['PRES', 'Ir']].values[2000:2150,:]
X = np.hstack([X, np.ones((X.shape[0],1))])
Y = data['TEMP'].values[2000:2150]
# Use Linear Algebra to solve
a = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
predictedY = np.dot(X, a)
x = np.array([X[:, 0], X[:,2]]).T
y = np.dot(x, np.linalg.solve(np.dot(x.T, x), np.dot(x.T, Y)))

#===============
# First subplot
#===============
# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=(15,5))
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1)
# plot a scatters points
ax.scatter(X[:, 0], Y)
ax.plot(x[:,0],y, c='r')
ax.set_title(r'Linear regression: $T = \omega_{0} + \omega_{1}P + \epsilon $')
ax.set_xlabel('Pressure (P)')
ax.set_ylabel('Temperature (T)')

#===============
# Second subplot
#===============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
# plot a 3D wireframe like in the example mplot3d/wire3d_demo

# create a wiremesh for the plane that the predicted values will lie
xx, yy, zz = np.meshgrid(X[:, 0], X[:, 1], X[:, 2])
combinedArrays = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
Z = combinedArrays.dot(a)

ax.scatter(X[:, 0], X[:, 1], Y)
ax.set_title(r'Multiple linear regression: $T = \omega_{0} + \omega_{1}P + \omega_{2}Ir + \epsilon $')
ax.set_xlabel('Pressure (P)')
ax.set_ylabel('Cumulated hours of rain (Ir)')
ax.set_zlabel('Temperature (T)')
ax.plot_trisurf(combinedArrays[:, 0], combinedArrays[:, 1], Z, alpha=0.5, color='r')
plt.show()