import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv",dtype = float, delimiter = ",")
data = np.transpose(data)

# time axis
x = data[0]
x[0] = 0

# cavity voltage
v = -data[2]

# Lorentzian function
def lorentzian(x, A, x0, y, B):
    return A / (np.pi * y * (1 + ((x - x0) / y) ** 2)) + B

# Example data (x_data and y_data should be your actual data)

# Fit the data to the Lorentzian function
popt, pcov = curve_fit(lorentzian, x, v, p0=[100, 600, 50, 0])

# popt contains the best-fit values for A, x0, y, B
print("Best-fit parameters:", popt)

fitted_y_data = lorentzian(x, *popt)

# Plot the data and the fit
plt.figure(figsize=(8, 6))
plt.scatter(x, v, label="Noisy Data", color='red', marker='o')  # Noisy data points
plt.plot(x, fitted_y_data, label="Fitted Lorentzian", color='blue', linewidth=2)  # Fitted curve
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Lorentzian Fit to Data')
plt.show()

