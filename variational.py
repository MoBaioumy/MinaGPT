import numpy as np
import matplotlib.pyplot as plt

# Generate some data: 100 points from a normal distribution with mean=0 and variance=1
true_mu = 0
sigma_squared = 1
n_samples = 100
data = np.random.normal(true_mu, np.sqrt(sigma_squared), n_samples)

# Initialize variational parameters for q(mu) - our approximation to the posterior
mu_q = 1.0  # initial guess for the mean
sigma_q_squared = 1.0  # initial guess for the variance

learning_rate = 0.01
iterations = 1000

for i in range(iterations):
    # Compute gradients for mu_q and sigma_q_squared
    # Note: These gradients are derived from the ELBO (Evidence Lower BOund) calculation
    grad_mu_q = np.sum(data - mu_q) / sigma_q_squared - mu_q / sigma_squared
    grad_sigma_q_squared = 0.5 * (np.sum((data - mu_q)**2) / sigma_q_squared - n_samples) / sigma_q_squared - 0.5 / sigma_squared
    
    # Update variational parameters
    mu_q += learning_rate * grad_mu_q
    sigma_q_squared += learning_rate * grad_sigma_q_squared

print(f"Estimated mean (mu_q): {mu_q}")
print(f"Estimated variance (sigma_q_squared): {sigma_q_squared}")

# Visualizing the true distribution and our approximation
plt.hist(data, bins=30, alpha=0.5, label='Data')
x = np.linspace(min(data), max(data), 100)
plt.plot(x, np.exp(-(x - true_mu)**2 / (2 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared), label='True Distribution')
plt.plot(x, np.exp(-(x - mu_q)**2 / (2 * sigma_q_squared)) / np.sqrt(2 * np.pi * sigma_q_squared), label='Approximation')
plt.legend()
plt.show()

# Visualizing the true distribution and our approximation
plt.hist(data, bins=30, alpha=0.5, label='Data')
x = np.linspace(min(data), max(data), 100)
plt.plot(x, np.exp(-(x - true_mu)**2 / (2 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared), label='True Distribution')
plt.plot(x, np.exp(-(x - mu_q)**2 / (2 * sigma_q_squared)) / np.sqrt(2 * np.pi * sigma_q_squared), label='Approximation')
plt.legend()

# Save the figure before showing it
plt.savefig('result.png')

plt.show()

