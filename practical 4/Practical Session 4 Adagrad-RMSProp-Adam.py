#!/usr/bin/env python
# coding: utf-8

# ## Practical Work 4

# For this practical work, the student will have to develop a Python program that is able to implement the accelerated gradient descent methods with adaptive learning rate <b>(Adagrad, RMSProp, and Adam)</b> in order to achieve the linear regression of a set of datapoints.

# #### Import numpy, matplotlib.pyplot and make it inline

# In[529]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import r2_score
import math
from numpy import linalg as LA


# To have a dataset or set of data points, the student must generate a pair of arrays <b>X</b> and <b>y</b> with the values in <b>X</b> equally distributed between <b>0</b> and <b>20</b> and the values in <b>y</b> such that: 
# <b>yi = a*xi + b (and a = -1, b = 2)</b>
# 

# In[530]:


x = np.linspace(0, 20,50)
y = -1 * x + 2


# In[531]:


print(x,y)


# #### Plot your data points. 

# In[532]:


plt.scatter(x,y)


# ## Adagrad

# ### For a single variable linear regression ML model, build a function to find the optimum Theta_0 and Theta_1 parameters using Adagrad optimization algorithm.
# #### The funtion should have the following input parameters:
# ##### 1. Input data as a matrix (or vector based on your data).
# ##### 2. Target label as a vector.
# ##### 3. Learning rate.
# ##### 4. Epsilon.
# ##### 5. Maximum number of iterations (Epochs).
# #### The funtion should return the following outputs:
# ##### 1. All predicted Theta_0 in all iterations.
# ##### 2. All predicted Theta_1 in all iterations.
# ##### 3. Corresponding loss for each Theta_0 and Theta_1 predictions.
# ##### 4.All hypothesis outputs (prdicted labels) for each Theta_0 and Theta_1 predictions.
# ##### 5.Final Optimum values of Theta_0 and Theta_1.
# #### Choose the suitable number of iterations, learning rate, Epsilon, and stop criteria.
# #### Calculate r2 score. Shouldn't below 0.9
# #### Plot the required curves (loss-epochs, loss-theta0, loss-theta1, all fitted lines per epoch (single graph), best fit line)
# #### Try different values of the huperparameters and see the differnce in your results.

# ![image.png](attachment:image.png)

# In[533]:


#### Adagrad Algorithm
def adagrad_algorithm(x,y,learning_rate,epsilon,max_iter):
        final_thetas = []
        cost_func = [];
        theta0_val = [];
        theta1_val = [];
        hypothesis_output = [];
        theta_0 = 0
        theta_1 = 0
        vt_zero = 0
        vt_one = 0
        i=1
        while i <= max_iter :
            #step 2
            output_hx = theta_0 + theta_1 * x
            hypothesis_output.append(output_hx)
            #step 3
            j_theta = 0
            j_theta = (1/(2*output_hx.size))*((output_hx - y)**2).sum()
            cost_func.append(j_theta)
            theta0_val.append(theta_0)
            theta1_val.append(theta_1)
            #step 4
            #theta 0 gradient
            theta_0_gradient = 0
            theta_0_gradient = (1/(output_hx.size)) * (output_hx - y).sum()
            #theta 1 gradient
            theta_1_gradient = 0
            theta_1_gradient = (1/(output_hx.size)) * ( (output_hx - y) * x ).sum()
            #step 5
            vt_zero = vt_zero + theta_0_gradient**2
            vt_one = vt_one + theta_1_gradient**2
            #next theta 0 => update
            theta_0 = theta_0 - (learning_rate / ( math.sqrt(vt_zero) + epsilon ))* theta_0_gradient
            #next theta 1 => update
            theta_1 = theta_1 - (learning_rate / ( math.sqrt(vt_one) + epsilon ))* theta_1_gradient
            gradient_vector = np.array(theta_0_gradient,theta_1_gradient)
            gradient_vector_norm = LA.norm(gradient_vector)
            if i == max_iter or gradient_vector_norm<=0.0001:#stop criteria
                final_theta_0 = theta_0
                final_theta_1 = theta_1
                final_thetas.append(final_theta_0)
                final_thetas.append(final_theta_1)
                return final_thetas,cost_func,theta0_val,theta1_val,hypothesis_output
            i+=1


# In[534]:


final_thetas,cost_func,theta0_val,theta1_val,hypothesis_output = adagrad_algorithm(x,y,0.15,1e-8,1000)
print("No of iterations: " , len(cost_func))


# In[535]:


print(r2_score(y, hypothesis_output[-1]))


# In[536]:


plt.plot(range(len(cost_func)),cost_func)
plt.xlabel("epochs")
plt.ylabel("cost func")


# In[537]:


plt.plot(theta0_val,cost_func)
plt.xlabel("theta zero")
plt.ylabel("cost func")


# In[538]:


plt.plot(theta1_val,cost_func)
plt.xlabel("theta one")
plt.ylabel("cost func")


# In[539]:


for i in range(len(theta0_val)):
    plt.plot(x,hypothesis_output[i])


# In[540]:


plt.scatter(x,y)
plt.plot(x,hypothesis_output[-1],color="red")


# ## Trying Different Hyperparameter

# In[541]:


final_thetas,cost_func,theta0_val,theta1_val,hypothesis_output = adagrad_algorithm(x,y,0.08,1e-8,1000)
print("No of iterations: " , len(cost_func))
print(r2_score(y, hypothesis_output[-1]))


# In[542]:


plt.plot(range(len(cost_func)),cost_func)
plt.xlabel("epochs")
plt.ylabel("cost func")


# In[543]:


plt.plot(theta0_val,cost_func)
plt.xlabel("theta zero")
plt.ylabel("cost func")


# In[544]:


plt.plot(theta1_val,cost_func)
plt.xlabel("theta one")
plt.ylabel("cost func")


# In[545]:


for i in range(len(theta0_val)):
    plt.plot(x,hypothesis_output[i])


# In[546]:


plt.scatter(x,y)
plt.plot(x,hypothesis_output[-1],color="red")


# ## RMSProp

# ### Update the previos implementation to be RMSProp.
# #### Compare your results with Adagrad results.

# ![image.png](attachment:image.png)

# In[547]:


#### RMSProp  Algorithm
def RMSProp_algorithm(x,y,learning_rate,epsilon,max_iter):
        beta = 0.9
        final_thetas = []
        cost_func = [];
        theta0_val = [];
        theta1_val = [];
        hypothesis_output = [];
        theta_0 = 0
        theta_1 = 0
        vt_zero = 0
        vt_one = 0
        i=1
        while i <= max_iter :
            #step 2
            output_hx = theta_0 + theta_1 * x
            hypothesis_output.append(output_hx)
            #step 3
            j_theta = 0
            j_theta = (1/(2*output_hx.size))*((output_hx - y)**2).sum()
            cost_func.append(j_theta)
            theta0_val.append(theta_0)
            theta1_val.append(theta_1)
            #step 4
            #theta 0 gradient
            theta_0_gradient = 0
            theta_0_gradient = (1/(output_hx.size)) * (output_hx - y).sum()
            #theta 1 gradient
            theta_1_gradient = 0
            theta_1_gradient = (1/(output_hx.size)) * ( (output_hx - y) * x ).sum()
            #step 5
            vt_zero = beta * vt_zero + (1-beta) * theta_0_gradient**2
            vt_one = beta * vt_one + (1-beta) * theta_1_gradient**2
            #next theta 0 => update
            theta_0 = theta_0 - (learning_rate / ( math.sqrt(vt_zero) + epsilon ))* theta_0_gradient
            #next theta 1 => update
            theta_1 = theta_1 - (learning_rate / ( math.sqrt(vt_one) + epsilon ))* theta_1_gradient
            gradient_vector = np.array(theta_0_gradient,theta_1_gradient)
            gradient_vector_norm = LA.norm(gradient_vector)
            if i == max_iter or gradient_vector_norm<=0.0001:#stop criteria
                final_theta_0 = theta_0
                final_theta_1 = theta_1
                final_thetas.append(final_theta_0)
                final_thetas.append(final_theta_1)
                return final_thetas,cost_func,theta0_val,theta1_val,hypothesis_output
            i+=1


# In[548]:


final_thetas,cost_func,theta0_val,theta1_val,hypothesis_output = RMSProp_algorithm(x,y,0.01,1e-8,400)
print("No of iterations: " , len(cost_func))


# In[549]:


print(r2_score(y, hypothesis_output[-1]))


# In[550]:


plt.plot(range(len(cost_func)),cost_func)
plt.xlabel("epochs")
plt.ylabel("cost func")


# In[551]:


plt.plot(theta0_val,cost_func)
plt.xlabel("theta zero")
plt.ylabel("cost func")


# In[552]:


plt.plot(theta1_val,cost_func)
plt.xlabel("theta one")
plt.ylabel("cost func")


# In[553]:


for i in range(len(theta0_val)):
    plt.plot(x,hypothesis_output[i])


# In[554]:


plt.scatter(x,y)
plt.plot(x,hypothesis_output[-1],color="red")


# ## Adam

# ### Update the previos implementation to be Adam.
# #### Compare your results with Adagrad and RMSProp results.

# ![image-4.png](attachment:image-4.png)

# In[555]:


#### Adam  Algorithm
def Adam_algorithm(x,y,learning_rate,epsilon,max_iter):
        beta1 = 0.9
        beta2 = 0.99
        final_thetas = []
        cost_func = [];
        theta0_val = [];
        theta1_val = [];
        hypothesis_output = [];
        theta_0 = 0
        theta_1 = 0
        vt_zero = 0
        vt_one = 0
        mt_zero = 0
        mt_one = 0
        i=1
        while i <= max_iter :
            #step 2
            output_hx = theta_0 + theta_1 * x
            hypothesis_output.append(output_hx)
            #step 3
            j_theta = 0
            j_theta = (1/(2*output_hx.size))*((output_hx - y)**2).sum()
            cost_func.append(j_theta)
            theta0_val.append(theta_0)
            theta1_val.append(theta_1)
            #step 4
            #theta 0 gradient
            theta_0_gradient = 0
            theta_0_gradient = (1/(output_hx.size)) * (output_hx - y).sum()
            #theta 1 gradient
            theta_1_gradient = 0
            theta_1_gradient = (1/(output_hx.size)) * ( (output_hx - y) * x ).sum()
            #step 5
            mt_zero = beta1 * mt_zero + (1-beta1) * theta_0_gradient
            vt_zero = beta2 * vt_zero + (1-beta2) * theta_0_gradient**2
            #correction
            mt_zero_corrected = mt_zero / (1-beta1**i)
            vt_zero_corrected = vt_zero / (1-beta2**i)
            
            mt_one = beta1 * mt_one + (1-beta1) * theta_1_gradient
            vt_one = beta2 * vt_one + (1-beta2) * theta_1_gradient**2
            #correction
            mt_one_corrected = mt_one / (1-beta1**i)
            vt_one_corrected = vt_one / (1-beta2**i)
            #next theta 0 => update
            theta_0 = theta_0 - (learning_rate / ( math.sqrt(vt_zero_corrected) + epsilon ))* mt_zero_corrected
            #next theta 1 => update
            theta_1 = theta_1 - (learning_rate / ( math.sqrt(vt_one_corrected) + epsilon ))* mt_one_corrected
            gradient_vector = np.array(theta_0_gradient,theta_1_gradient)
            gradient_vector_norm = LA.norm(gradient_vector)
            if i == max_iter or gradient_vector_norm<=0.0001:#stop criteria
                final_theta_0 = theta_0
                final_theta_1 = theta_1
                final_thetas.append(final_theta_0)
                final_thetas.append(final_theta_1)
                return final_thetas,cost_func,theta0_val,theta1_val,hypothesis_output
            i+=1


# In[556]:


final_thetas,cost_func,theta0_val,theta1_val,hypothesis_output = Adam_algorithm(x,y,0.01,1e-8,1000)
print("No of iterations: " , len(cost_func))


# In[557]:


print(r2_score(y, hypothesis_output[-1]))


# In[558]:


plt.plot(range(len(cost_func)),cost_func)
plt.xlabel("epochs")
plt.ylabel("cost func")


# In[559]:


plt.plot(theta0_val,cost_func)
plt.xlabel("theta zero")
plt.ylabel("cost func")


# In[560]:


plt.plot(theta1_val,cost_func)
plt.xlabel("theta one")
plt.ylabel("cost func")


# In[561]:


for i in range(len(theta0_val)):
    plt.plot(x,hypothesis_output[i])


# In[562]:


plt.scatter(x,y)
plt.plot(x,hypothesis_output[-1],color="red")


# ## Congratulations 
# ![image.png](attachment:image.png)
