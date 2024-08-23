import numpy as np
import matplotlib.pyplot as plt

# Define the input parameters
r1 = 0.98 # Reflectance of mirror 1 
r2 = 0.98 # Reflectance of mirror 2 
t1 = 0.98 # Transmittance of mirror 1 
t2 = 0.98 # Transmittance of mirror 2 
n = 1
wavelen = 1550e-8
k = 2 * np.pi * n / wavelen 
A0 = 100 # Input power mW
# Define a range of cavity lengths L 
L = np.linspace(0, 2* wavelen, 10000)  # From 0 to 5 wavelengths)

# Define the transfer matrix for mirror 1 (M1)
M1 = (1/(1j*t1))*np.array([
    [-1, -r1],
    [r1, 1]
])

# Define the propagation matrix (PLn)
PLn = np.array([
    [np.exp(1j * k * L), 0],
    [0, np.exp(-1j * k * L)]
])

# Define the transfer matrix for mirror 2 (M2)
M2 = (1/(1j*t2))*np.array([
    [-1, -r2],
    [r2, 1]
])

# Calculate the cavity transfer matrix (C)
C = np.dot(M2, np.dot(PLn, M1))

# Print the elements of the cavity transfer matrix C
C11, C12 = C[0, 0], C[0, 1]
C21, C22 = C[1, 0], C[1, 1]

# print("Cavity Transfer Matrix C:")
# print(f"C11 = {C11}")
# print(f"C12 = {C12}")
# print(f"C21 = {C21}")
# print(f"C22 = {C22}")

# Define cavity reflection coefficient
r = -C21/C22

# Field Transmission Coefficient
t = (C11*C22-C21*C12)/C22

def power_ratio(t1, r1,r2,L, k):
    # different ways to calculate power ratio to verify
    power_ratio = 1j/t1*(1+r1*r)
    # power_ratio = 1j*t1 / (1-r1*r2*np.exp(2j * k * L)) 
    power_ratio_mag = power_ratio.real**2 + power_ratio.imag**2
    return  power_ratio_mag
    # return (t1**2) / (1 + r1**2 * r2**2 - 2 * r1 * r2 * np.cos(2 * k * L))
    
# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(L / wavelen, power_ratio(t1, r1, r2, k, L), label=r'$\left(\frac{A_1}{A_0}\right)^2$')
plt.xlabel('Cavity Length (L / λ)')
plt.ylabel('Power Ratio (A1 / A0)^2')
plt.title('Cavity Power Enhancement vs Cavity Length')
plt.grid(True)
plt.legend()
plt.show()