#!/usr/bin/env python
# coding: utf-8

# ## Practical Work 1

# For this practical work, the student will have to develop a Python program that is able to implement the gradient descent in order to achieve the linear regression of a set of datapoints.

# #### Import numpy, matplotlib.pyplot and make it inline

# In[159]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Read RegData csv file into numpy array  (check your data)
# ##### Data source
# https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html
# 

# In[160]:


from numpy import genfromtxt
my_data = genfromtxt('RegData.csv', delimiter=',')


# #### Explore your data

# In[161]:


my_data


# #### Define variables X and y. Assign first column data to X and second column to y
# <b>Note:</b> X is the independent variable (input to LR model) and y is the dependent variable (output)

# In[162]:


x = my_data[:, 0];
y = my_data[:, 1];


# #### Explore your data

# In[163]:


x


# In[164]:


y


# #### Plot the original data (scatter plot of X,y)

# In[165]:


plt.scatter(x, y, marker='o');


# ## LR Full Implementation

# ### Step1: Initialize parameters (theta_0 & theta_1) with random value or simply zero. Also choose the Learning rate. 

# ![image.png](attachment:image.png)

# In[166]:


theta_0 = 0;
theta_1 = 0;
learning_rate = 0.01;


# ### Step2: Use (theta_0 & theta_1) to predict the output h(x)= theta_0 + theta_1 * x.![image.png](attachment:image.png)
# #### Note: you will need to iterate through all data points

# In[167]:


output_hx = theta_0 + theta_1 * x;
print(output_hx);


# ### Step3: Calculate Cost function 𝑱(theta_0,theta_1 ).![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[168]:


j_theta = 0;

for i in range(output_hx.size):
  j_theta += (output_hx[i] - y[i]) * (output_hx[i] - y[i]);
j_theta *= 1/(2*output_hx.size);

print(j_theta);


# ### Step4: Calculate the gradient.![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[169]:


#theta 0 gradient
theta_0_gradient = 0;
for i in range(output_hx.size):
  theta_0_gradient += (output_hx[i] - y[i]);
theta_0_gradient *= 1/(output_hx.size);
print(theta_0_gradient);

#theta 1 gradient
theta_1_gradient = 0;
for i in range(output_hx.size):
  theta_1_gradient += (output_hx[i] - y[i]) * x[i];
theta_1_gradient *= 1/(output_hx.size);
print(theta_1_gradient);


# ### Step5: Update the parameters (simultaneously).![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# In[170]:


#next theta 0
next_theta_0 = theta_0 - learning_rate * theta_0_gradient;
print(next_theta_0);

#next theta 1
next_theta_1 = theta_1 - learning_rate * theta_1_gradient;
print(next_theta_1);


# ### Step6: Repeat from 2 to 5 until converge to the minimum or achieve maximum iterations.![image.png](attachment:image.png)

# In[171]:


final_thetas = [];
cost_func = [];
def gradient_descent(theta_0,theta_1,iter):
        #step 2
        output_hx = theta_0 + theta_1 * x;
        #step 3
        j_theta = 0;
        for i in range(output_hx.size):
          j_theta += (output_hx[i] - y[i]) * (output_hx[i] - y[i]);
        j_theta *= 1/(2*output_hx.size);
        cost_func.append(j_theta);
        #step 4
        #theta 0 gradient
        theta_0_gradient = 0;
        for i in range(output_hx.size):
          theta_0_gradient += (output_hx[i] - y[i]);
        theta_0_gradient *= 1/(output_hx.size);
        #theta 1 gradient
        theta_1_gradient = 0;
        for i in range(output_hx.size):
          theta_1_gradient += (output_hx[i] - y[i]) * x[i];
        theta_1_gradient *= 1/(output_hx.size);
        #step 5
        #next theta 0
        next_theta_0 = theta_0 - learning_rate * theta_0_gradient;
        #next theta 1
        next_theta_1 = theta_1 - learning_rate * theta_1_gradient;
        
        if( iter == 1000 or round(next_theta_0,5) == round(theta_0,5) ):
            final_theta_0 = theta_0;
            final_theta_1 = theta_1;
            final_thetas.append(final_theta_0);
            final_thetas.append(final_theta_1);
            final_predict = output_hx;
            print("final");
            print(final_theta_0);
            print(final_theta_1);
            print(output_hx);
            print(y);
        else:
            iter += 1;
            gradient_descent(next_theta_0,next_theta_1,iter)
        
gradient_descent(0,0,0);
print(final_thetas);


# #### Predict y values using the LR equation
# ##### h(x)= theta_0 + theta_1 * x

# In[172]:


h_predict = final_thetas[0] + final_thetas[1] * x;


# #### Plot  LR equation output (fitted line) with the original data (scatter plot of X,y)

# In[173]:


plt.scatter(x,y)
plt.plot(x,h_predict)


# #### Use R2 score to evaluate LR equation output
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)
# ![image-3.png](attachment:image-3.png)
# https://en.wikipedia.org/wiki/Coefficient_of_determination

# In[174]:


#Using r2 score from sklearn.metrics to evaluate prediction performance.
from sklearn.metrics import r2_score
print(r2_score(y, h_predict))


# ## GD vectorize Implementation
# ### Implement GD without iterate through data points i.e. use vector operations

# In[175]:


final_thetas = [];
cost_func = [];
def gradient_descent_vectorize(theta_0,theta_1,iter):
        #step 2
        output_hx = theta_0 + theta_1 * x;
        #step 3
        j_theta = 0;
        j_theta = (1/(2*output_hx.size))*((output_hx - y)**2).sum();
        cost_func.append(j_theta);
        #step 4
        #theta 0 gradient
        theta_0_gradient = 0;
        theta_0_gradient = (1/(output_hx.size)) * (output_hx - y).sum();
        #theta 1 gradient
        theta_1_gradient = 0;
        theta_1_gradient = (1/(output_hx.size)) * ( (output_hx - y) * x ).sum();
        #step 5
        #next theta 0
        next_theta_0 = theta_0 - learning_rate * theta_0_gradient;
        #next theta 1
        next_theta_1 = theta_1 - learning_rate * theta_1_gradient;
        
        if(iter == 1000 or round(next_theta_0,5) == round(theta_0,5) ):
            final_theta_0 = theta_0;
            final_theta_1 = theta_1;
            final_thetas.append(final_theta_0);
            final_thetas.append(final_theta_1);
            final_predict = output_hx;
            print("final");
            print(final_theta_0);
            print(final_theta_1);
            print(output_hx);
        else:
            iter += 1;
            gradient_descent_vectorize(next_theta_0,next_theta_1,iter)
    
gradient_descent_vectorize(0,0,0);
print(final_thetas);


# #### Plot the output and calculate R2 score
# ##### Make sure that you obtained the same results

# In[176]:


h_predict = final_thetas[0] + final_thetas[1] * x;


# In[177]:


plt.scatter(x,y)
plt.plot(x,h_predict)


# In[178]:


from sklearn.metrics import r2_score
print(r2_score(y, h_predict))


# ## Plot loss function

# ### Plot loss vs. iterations

# In[179]:


plt.plot(cost_func);


# ## Multivariate LR

# #### Read MultipleLR csv file into numpy array  (check your data)
# ##### Data source
# https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/frame.html
# 

# In[180]:


my_data_multi = genfromtxt('MultipleLR.csv', delimiter=',');
my_data_multi


# In[181]:


num_rows, num_cols = my_data_multi.shape;
x0 = np.ones(num_rows);
b=x0.reshape(-1,1);
data_array = np.concatenate((b, my_data_multi), axis=1);
print(data_array)


# In[182]:


#actual output => last column of data
actual_y = data_array[:,4];
print(actual_y);


# In[183]:


#actual inputs => x1 , x2 , x3
x0 = data_array[:,0];
x1 = data_array[:,1];
x2 = data_array[:,2];
x3 = data_array[:,3];
print(x0);
print(x1);
print(x2);
print(x3);


# In[184]:


#input data array => x0 , x1 , x2 , x3
input_data_array = data_array[:,[0,1,2,3]];
print(input_data_array);
xs_array = data_array[:,[1,2,3]];
print(xs_array);


# In[185]:


#gradient descent for multi variable
final_thetas_multi = [];
cost_func_multi = [];
learning_rate = 0.00001;
def gradient_descent_multi(theta_0,theta_1,theta_2,theta_3,iter):
    theta_array = np.array( [theta_0,theta_1,theta_2,theta_3] );
    predict_out = np.dot(input_data_array, theta_array);
    #cost function
    j_theta = 0;
    j_theta = (1/(2*predict_out.size))*((predict_out - actual_y)**2).sum();
    cost_func_multi.append(j_theta);
    #theta 0 gradient
    theta_0_gradient = 0;
    theta_0_gradient = (1/(predict_out.size)) * ((predict_out-actual_y).sum());
        
    #theta 1 gradient
    theta_1_gradient = 0;
    theta_1_gradient = (1/(predict_out.size)) * ((predict_out-actual_y)*x1).sum();

    #theta 2 gradient
    theta_2_gradient = 0;
    theta_2_gradient = (1/(predict_out.size)) * ((predict_out-actual_y)*x2).sum();

    #theta 3 gradient
    theta_3_gradient = 0;
    theta_3_gradient = (1/(predict_out.size)) * ((predict_out-actual_y)*x3).sum();
    
    #next theta 0
    next_theta_0 = theta_0 - learning_rate * theta_0_gradient;

    #next theta 1
    next_theta_1 = theta_1 - learning_rate * theta_1_gradient;

    #next theta 2
    next_theta_2 = theta_2 - learning_rate * theta_2_gradient;

    #next theta 3
    next_theta_3 = theta_3 - learning_rate * theta_3_gradient;
    
    if(iter == 100 ):
        final_theta_0 = theta_0;
        final_theta_1 = theta_1;
        final_theta_2 = theta_2;
        final_theta_3 = theta_3;
            
        final_thetas_multi.append(final_theta_0);
        final_thetas_multi.append(final_theta_1);
        final_thetas_multi.append(final_theta_2);
        final_thetas_multi.append(final_theta_3);
    else:
        iter += 1;
        gradient_descent_multi(next_theta_0,next_theta_1,next_theta_2,next_theta_3,iter);

gradient_descent_multi(0,0,0,0,0);
print("------- result--------");
print(final_thetas_multi);
#print(cost_func_multi);
final_predict = np.dot(input_data_array, final_thetas_multi);
print(final_predict);
print(actual_y);


# In[186]:


#### Use R2 score to evaluate LR equation output


# In[187]:


print(r2_score(actual_y, final_predict));


# In[188]:


### Plot loss vs. iterations


# In[189]:


plt.plot(cost_func_multi);


# # Bonus
# ## LR Using sklearn

# ### Single Variable

# #### Build a LR model usin linearmodel.LinearRegression() from sklearn library

# In[190]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd


# In[191]:


dataset = pd.read_csv('RegData.csv')


# In[192]:


dataset.shape


# In[193]:


X = dataset.iloc[:, :-1].values #all except last column
y = dataset.iloc[:, 1].values #first column


# In[194]:


#80% training set #20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# #### Train the model (fit the model to the training data)

# In[195]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[196]:


#intercept
print(regressor.intercept_)


# In[197]:


#slope
print(regressor.coef_)


# #### Predict y values using the trained model

# In[198]:


#predict y
y_pred = regressor.predict(X_test)


# In[199]:


#compare result
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# #### Plot model output (fitted line) with the original data (scatter plot of X,y)

# In[200]:


plt.scatter(X,y)
plt.plot(X_test,y_pred)


# In[201]:


plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)


# #### Use R2 score to evaluate model output

# In[202]:


r2_score(y_test, y_pred)


# ### Repeat for Mulivariate

# In[203]:


dataset = pd.read_csv('MultipleLR.csv')


# In[204]:


X = dataset[['73', '80', '75']] #first 3 columns
y = dataset['152'] #last column


# In[205]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[206]:


#train the algorithm
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[207]:


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[208]:


#predict
y_pred = regressor.predict(X_test)


# In[209]:


#compare
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[210]:


r2_score(y_test, y_pred)

