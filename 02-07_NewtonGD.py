#newton's method for optimization
# Newton's method is an iterative optimization algorithm used to find the minimum or maximum of a function.
# It uses the first and second derivatives of the function to find the optimal point.
# The algorithm starts with an initial guess and iteratively updates the guess using the formula:
# x_new = x_old - f'(x_old) / f''(x_old)
# The algorithm converges to the optimal point when the change in x is small enough.
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
def f(x):     
    return(-np.cos(np.pi*x/20)+x**2) 
def df(x):     
     return(np.sin(np.pi*x/20)*np.pi/20+2*x) 
def ddf(x):     
     return((np.pi/20)**2*np.cos(np.pi*x/20)+2)
def Newton(start, iterations):     
    x = start     
    Newton_x, Newton_y = [], []     
    for it in range(iterations):         
        Newton_x.append(x), Newton_y.append(f(x))         
        g = df(x)         
        h = ddf(x) 
        x = x - g/h
    return(Newton_x,Newton_y) 
def plot_track(iterations):     
    Newton_x, Newton_y = Newton( start = -20, iterations = iterations)    
    points_x = np.linspace(-20, 20, 100)     
    points_y = f(points_x)     
    sns.set_theme(style='white')     
    plt.plot(points_x,points_y,c="k",alpha= 1, linestyle="-", linewidth= 2)     
    plt.scatter(Newton_x, Newton_y, c = 'red', alpha = 0.8, s = 20)     
    u =np.array([Newton_x[i+1]-Newton_x[i] for i in range(len(Newton_x)-1)])
    v=np.array([Newton_y[i+1]- Newton_y[i] for i in range(len(Newton_y)-1)])
    plt.quiver(Newton_x[:len(u)], Newton_y[:len(v)], u, v, angles='xy', width= 0.005, scale_units='xy', scale =1 ,alpha=0.9, color = 'gray')    
    plt.xlabel('x')
    plt.ylabel('$L')
    plt.title('iterations:{}'.format(iterations))     
    plt.show() 
plot_track(20)
