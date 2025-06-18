import numpy as np
import matplotlib.pyplot as plt
from saw_elastic_predictions.materials import Material
from saw_elastic_predictions.saw_calculator import SAWCalculator

# Create Ni3Al material (at 500C)
ni3al = Material(
    formula='Ni3Al',
    C11=150.4e9,  # Pa
    C12=81.7e9,   # Pa
    C44=107.8e9,  # Pa
    density=7.57e3,  # kg/m^3
    crystal_class='cubic'
)

# Euler angles for {110}<111> orientation (in radians)
# This replicates euler_110_111 from the MATLAB euler_database
euler_110_111 = np.array([2.186, 0.9553, 2.186])  # These values need verification

# Calculate SAW speeds for angles 0 to 60 degrees
angles = np.arange(0, 61)
saw_speeds = np.zeros((2, len(angles)))  # For both SAW and PSAW

# Initialize calculator
calculator = SAWCalculator(ni3al, euler_110_111)

# Calculate speeds for each angle
for i, angle in enumerate(angles):
    print(f"Calculating angle {angle}")
    v, _, _ = calculator.get_saw_speed(angle, sampling=4000, psaw=1)
    saw_speeds[:len(v), i] = v

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(angles, saw_speeds[0, :], 'k-', linewidth=1.25)
plt.xlabel('Relative surface angle [degrees]', fontsize=20)
plt.ylabel('SAW speed [m/s]', fontsize=20)
plt.title('Ni3Al SAW speeds on (111)', fontsize=16)

# Set font properties
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().set_box_aspect(1)  # Make plot square
#plt.gca().set_linewidth(1.25)

plt.tight_layout()
plt.show() 