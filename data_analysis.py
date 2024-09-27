import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import spinmob as sp

n = 1
wavelen = 1550
k = 2 * np.pi * n / wavelen 

x = np.linspace(0, wavelen*5, 10000)  # From 0 to 5 wavelengths
def power_function(x, r, w, t_0, y_0, A_0):
    return (2*r**2 - 2*r**2*np.cos(2*w*(x-t_0)))/(1 + r**4 - 2*r**2*np.cos(2*w*(x-t_0)))*A_0 + y_0

# plt.plot(x, power_function(x, 0.99, 0.002, 200, 40, 40))
# plt.show()

data = np.genfromtxt("9_20_40Hz_2V_next.csv",dtype = float, delimiter = ",")
data = np.transpose(data)

# time axis
x = data[0]
x[0] = 0

# cavity voltage
v = data[1]

f = sp.data.fitter()
f(autoplot = True)
f(xlabel = 'time')
f(ylabel = 'voltage')
f.set_data(x, v, 0.005)
# power intensity function
f.set_functions(f='(2*r**2 - 2*r**2*cos(2*w*(x-t_0)))/(1 + r**4 - 2*r**2*cos(2*w*(x-t_0)))*A_0 + y_0', p='r=0.99, w=7000,t_0=5.94e-6, y_0=-0.3, A_0=0.3866')
#f.set_functions(f='(2*r**2 - 2*r**2*cos(2*w*(x-t_0)))/(1 + r**4 - 2*r**2*cos(2*w*(x-t_0)))*A_0 + y_0', p='r=0.99, w=0.0001,t_0=1, y_0=40, A_0=40')
f(xmin=0.000005)
f(xmax=0.000007)
f.fit()

f_l = sp.data.fitter()
f_l(autoplot = True)
f_l(xlabel = 'time')
f_l(ylabel = 'voltage')
f_l.set_data(x, v, 0.005)
# Lorentzian function:
f_l.set_functions(f='A*1/(3.1415926*y*(1+((x-x0)/y)**2))+B', p = 'y = 2.59e-7,x0 = 5.94e-6, A=3.173e-7 ,B=0')
f_l(xmin=0.000005)
f_l(xmax=0.000007)
f_l.fit()

result_power = f.get_fit_parameters()
power_uncertainty = f.get_fit_standard_errors()
r = result_power[0]
r_uncertaty = power_uncertainty[0]
print(r)
print('using power function to cualculate finesses:', 2*np.pi/(2-2*r**2))
print('Uncertainty: ', 2*np.pi/(2-2*r**2)**2*4*r*r_uncertaty)


result_lorentz = f_l.get_fit_parameters()
lorentz_uncertainty = f_l.get_fit_standard_errors()
y = result_lorentz[0]  
y_uncertainty = lorentz_uncertainty[0]
print(y)
d = 5200*40*2*y
print('Using Lorentzian function to calculate finesses:', (wavelen/2)/d)
print('Uncertainty: ', (wavelen/2)/(5200*40*2*y**2)*y_uncertainty)


