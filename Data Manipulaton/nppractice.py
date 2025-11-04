import numpy as np
import random
import matplotlib.pyplot as plt
# print(np.array([1,2,3]))
# random.seed(34) # ab results same rahege har baar same number generate hoga random lib use karne par
# each number given as input in the above function points to unique sequence
# print(random.randint(1,10))
# print(random.randint(1,10))
# print(np.random.rand(100,1)) # generates 100 values between 0 to 1

# in ML what we call bias is intercept in maths
# used to shift the line , see it just as straight line maths
# we experss bias as y=x.theta

np.random.seed(42)
x=2*np.random.rand(100,1)
y= 4 + 3*x + np.random.rand(100,1) # y= b +ax +noise

plt.scatter(x,y)
plt.title("Generated Data")
plt.show()

# adding bias to feature matrix

x_b=np.c_[np.ones((100,1)),x] 
# np.ones((100,1)) generates a matrix of 100x1 with all vlaue as 1

theta_best= np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T.dot(y))
#(θ = (XᵀX)⁻¹ Xᵀy ) 

print(theta_best)

# now we have found the optimal values of parameters which is exactly training of model as we have found the parameters by inputing the data in the model which suits best to reperesentation and now we can enter our new data that is x_new to make the predictions

x_new=np.array([[0],[1.5],[3]])
x_new_b=np.c_[np.ones((3,1)),x_new]

y_predict=x_new_b.dot(theta_best)
# print(x)
# print(y)
print(y_predict)
plt.plot(x_new,y_predict,color="red",label="prediction")
plt.scatter(x,y,label="data")
plt.legend()
plt.show()