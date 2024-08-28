from transfer_matrix import*

c = 299792458
A1 = power_ratio(t1,r1,r2,L,k)*A0
a = -1j*A1*np.sqrt(2*L/c)
a_mag = a.real**2 + a.imag**2

def decay_rate(c, tn, L):
    return c*tn**2/(2*L)

if __name__ == "__main__":
    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(L / wavelen, c/L*a_mag)
    
    plt.xlabel('Cavity Length (L / λ)')
    plt.ylabel('Circulating power')
    plt.title('Circulating power vs Cavity Length')
    plt.grid(True)
    plt.legend()
    plt.show()
