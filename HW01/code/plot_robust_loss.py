import numpy as np
import matplotlib.pyplot as plt

delta = 0.1
error = np.linspace(0,100,num=100)
loss_sqerr = error**2
loss_robust = (delta**2) * (np.sqrt((1 + error) / delta**2) - 1)
#plt.ylim([0,2])
plt.plot(error,loss_sqerr,label="Squared Error Loss")
plt.plot(error,loss_robust,label="Robust Loss")
plt.legend()
plt.show()