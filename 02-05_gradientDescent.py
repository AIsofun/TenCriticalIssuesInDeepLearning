# Gradient Descent
# Gradient Descent is an optimization algorithm used to minimize a function by iteratively moving towards the steepest descent as defined by the negative of the gradient.
# It is commonly used in machine learning and deep learning to minimize the loss function.
# The algorithm starts with an initial guess and iteratively updates the guess using the formula:
# x_new = x_old - learning_rate * f'(x_old)
# The learning_rate determines the step size of each iteration.
# If the learning rate is too small, the algorithm will converge slowly. If it is too large, the algorithm may overshoot the minimum and diverge.
# Gradient Descent is sensitive to the choice of learning rate and may require tuning to find the optimal value.
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
def f(x):   
    return(x**2) 
def df(x):   
    return(2*x) 
def GD(lr, start, iterations):   
    x = start  
    GD_x, GD_y = [], []   
    for it in range(iterations):     
        GD_x.append(x)
        GD_y.append(f(x))     
        dx = df(x)     
        x = x - lr * dx   
    return(GD_x,GD_y) 
def plot_track(learning_rate, iterations):     
    GD_x, GD_y=GD(lr = learning_rate, start=-20, iterations = iterations)    
    points_x=np.linspace(-20, 20, 100)     
    points_y = f(points_x)     
    sns.set(style='white')     
    plt.plot(points_x,points_y,c="k",alpha= 1, linestyle="-", linewidth = 2)     
    plt.scatter(GD_x, GD_y, c = 'red', alpha = 0.8, s = 20)  
    u = np.array([GD_x[i+1]- GD_x[i] for i in range(len(GD_x)-1)]) 
    v = np.array([GD_y[i+1]- GD_y[i] for i in range(len(GD_y)-1)])    
    plt.quiver(GD_x[:len(u)], GD_y[:len(v)], u, v, angles='xy', width= 0.005, scale_units='xy', scale =1 ,alpha=0.9, color = 'gray')    
    plt.xlabel('x')     
    plt.ylabel('$L')
    plt.title('learning rate:{} iterations:{}'.format(round(learning_rate,2), iterations))     
    plt.show() 
# plot_track(pow(2,-4.2)*16, 20)
plot_track(pow(2, -7)*16, 20)   