
#importing the required packages
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
%matplotlib inline


# function for loading the data ie (cat vs not_cat)
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


n_train=train_set_x_orig.shape[0]
n_test=test_set_x_orig.shape[0]
n_px=train_set_x_orig.shape[1]

#reshaping the training and test example
train_set_x_flatten = train_set_x_orig.reshape (train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape (test_set_x_orig.shape[0],-1).T


#standardizing our datasets 
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255



#sigmoid function
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s


#initialize_with_zeros function 
def initialize_with_zeros (dim):
    w=np.zeros((dim,1))
    b=0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w,b




# propagate function
def propagate(w,b,X,Y):
    m=X.shape[1]
    #forward propogation from X to cost function 
    A=sigmoid(np.dot(w.T,X)+b)
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    #backward propagation to find grad 
    dw=1/m*np.dot(X,(A-Y).T)
    db=1/m*np.sum(A-Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    return grads, cost

    




# optimize function
def optimize (w,b,X,Y,num_iterations,learning_rate, print_cost=False):
    costs=[]
    for i in range (num_iterations):
        grads,cost = propagate (w,b,X,Y) 
        dw=grads["dw"]
        db=grads["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        
        #recording of the costs 
        if i%100==0:
            costs.append(cost)
            
        #print the cost every 100 training iterations 
        if print_cost and i % 100 == 0:
            print ("Cost after iterations %i: %f" %(i,cost))
    params = {"w": w,
            "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# predict function

def predict (w,b, X):
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)
    
    for i in range (A.shape[1]):
        if A[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
            
    assert(Y_prediction.shape == (1,m))
    return Y_prediction




#model function
def model (X_train, Y_train,X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost= False):
    w,b=initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize (w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    #retriving the parameters w and b from dictionary "parameters"
    w=parameters["w"]
    b=parameters["b"]
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train= predict(w,b,X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

#running of the model we created
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate = 0.005, print_cost = True)



#plotting the learning curve 
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (after hundred each)')
plt.title ('graph showing the learning rate of our machine learning model')
plt.show()



#analysis on chosing of different values for learning rate
learning_rates= [0.01,0.001,0.0001,0.00001]
models = {}
for i in learning_rates:
    print ("The learning rate is :" +str(i))
    models[str(i)]=model (train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=3500, learning_rate = i , print_cost= False)
    print ('n'+"------------------------------------------------------------------->>>>>>>"+'\n')
for i in learning_rates:
    plt.plot (np.squeeze (models[str(i)]["costs"]),label =str(models[str(i)]["learning_rate"]))
plt.ylabel ('cost')
plt.xlabel ('iterations ( after 100 each)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()




