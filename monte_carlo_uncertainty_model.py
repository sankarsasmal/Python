
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
true_area = 223876
true_conc = 0.06
calib_mean = 2.25e-7
calib_std = 0.1e-7  # ±0.1e-7 uncertainty in calibration factor
noise_percent = 0.05
error_margin = 0.10
simulations = 10000

# --- Monte Carlo Simulation ---
np.random.seed(42)

# Sample calibration factors from a normal distribution
calib_factors = np.random.normal(loc=calib_mean, scale=calib_std, size=simulations)

# Simulate area measurements with uniform noise
simulated_areas = true_area * (1 + np.random.uniform(-noise_percent, noise_percent, simulations))

# Calculate predicted concentrations
predicted_concs = calib_factors * simulated_areas

# Define acceptable error bounds
lower_bound = true_conc * (1 - error_margin)
upper_bound = true_conc * (1 + error_margin)

# Compute probability of being within bounds
within_bounds = (predicted_concs >= lower_bound) & (predicted_concs <= upper_bound)
probability = np.mean(within_bounds)

# Print result
print(f"Probability of accurate prediction within ±{int(error_margin*100)}%: {probability:.2%}")

# Plot results
plt.figure(figsize=(8, 5))
plt.hist(predicted_concs, bins=50, color='skyblue', edgecolor='black')
plt.axvline(lower_bound, color='red', linestyle='--', label='10% Lower Bound')
plt.axvline(upper_bound, color='red', linestyle='--', label='10% Upper Bound')
plt.title('Monte Carlo Simulation with Calibration Factor Uncertainty')
plt.xlabel('Predicted Concentration')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
