import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
import seaborn as sns 
def f(x,y):     
     return x**2+y**2 
def partial_x(x,y):
     return 2*x 
def partial_y(y,x):
     return 2*y 
def GD(lr, start, iterations):     
    x, y= start[0],start[1]     
    GD_x, GD_y ,GD_z= [], [],[]     
    for it in range(iterations):        
         GD_x.append(x)         
         GD_y.append(y)         
         GD_z.append(f(x,y))         
         dx = partial_x(x,y)         
         dy = partial_y(y,x)         
         x = x - lr * dx         
         y = y - lr * dy     
    return(GD_x, GD_y, GD_z) 
def plot_track(learning_rate, iterations):     
     GD_x, GD_y, GD_z=GD(lr= learning_rate, start=[15,0.1], iterations = iterations)     
     a = np.linspace(-20,20,100)     
     b = np.linspace(-20,20,100)     
     A,B = np.meshgrid(a,b)     
     sns.set_theme(style = 'white')     
     plt.contourf(A, B, f(A,B), 10, alpha=0.8, cmap=cm.Greys)    
     plt.scatter(GD_x, GD_y, c = 'red', alpha = 0.8, s = 20)     
     u = np.array([GD_x[i+1]- GD_x[i] for i in range(len(GD_x)-1)])     
     v = np.array([GD_y[i+1]- GD_y[i] for i in range(len(GD_y)-1)])    
     plt.quiver(GD_x[:len(u)], GD_y[:len(v)], u, v, angles='xy', width=0.005, scale_units='xy', scale =1 ,alpha=0.9, color = 'k') 
     plt.xlabel('x')     
     plt.ylabel('y')
     plt.title('learning  rate:{} iterations:{}'.format(round(learning_rate,2),iterations))    
     plt.show() 
plot_track(learning_rate = pow(2, -7)*16, iterations = 100)