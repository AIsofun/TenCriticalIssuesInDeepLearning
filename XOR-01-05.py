import numpy as np 
def sigmoid(h):    
    return(1/(1+np.exp(-h))) 
def sigmoid_derivative(h):    
    return(sigmoid(h)*(1-sigmoid(h))) 
           
class NeuralNetwork(object):    
    def __init__(self,x,y):    
        self.input = x    
        self.weights=np.random.rand(x.shape[1],1)    
        self.weights1 = np.random.rand(1) 
        self.y  = y    
        self.output = np.zeros(self.y.shape) 
    def feedforward(self):    
        self.output = sigmoid(np.dot(self.input, self.weights)+self.weights1) 
    def backprop(self):
        d_weights=np.dot(self.input.T,(2*(self.y-self.output)*sigmoid_derivative(self.output)))    
        d_weights1=np.dot(2*(self.y-self.output).reshape(4),sigmoid_derivative(self.output).reshape(4))
        self.weights += d_weights 
        self.weights1 += d_weights1



import matplotlib.pyplot as plt 
import seaborn as sns 

#NAND
f=np.array([[0,0],[1,1],[0,1],[1,0]]) 
g=np.array([1,0,1,1]).reshape(4,1) 

#XOR
# f=np.array([[0,0],[1,1],[0,1],[1,0]]) 
# g=np.array([0,0,1,1]).reshape(4,1) 

NN=NeuralNetwork(f,g) 
mse=[] 
for i in range(100):    
    NN.feedforward()    
    NN.backprop()    
    g_pre=NN.output    
    m=(g-g_pre).flatten()    
    mse.append(np.dot(m,m)) 

sns.set_theme(style='white')
plt.plot(range(100),mse,'r-',label="train error") 
plt.title('Perceptron training')
plt.legend() 
plt.show()

plt.figure() 
#a=np.linspace(-0.1,0.55,100) 
a = np.linspace(-0.5, 1.5, 100)
for i in range(4):     
    if g[i][0]==0:       
        plt.scatter(f[i][0],f[i][1],s=200,c='b')     
    else:       
        plt.scatter(f[i][0],f[i][1],s=200,c='r') 
plt.title('NAND') 
w1,w2,w3=NN.weights[0],NN.weights[1],NN.weights1 
plt.plot(a,-(w1*a+w3)/w2,'r-',markersize=15) 

plt.show()