# sawbench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- TODO: Update license badge if needed -->

Python package for calculating Surface Acoustic Wave (SAW) properties based on elastic constants and crystal orientation.

## Installation

```bash
# Clone the repository
git clone https://github.com/mystaple/sawbench.git # TODO: Verify URL
cd sawbench

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

## Usage

See the `examples/` directory for usage scripts.

```python
# Example placeholder (from example_ni3al.py)
import numpy as np
from sawbench.materials import Material
from sawbench.saw_calculator import SAWCalculator

# Define material
material = Material(...)

# Define orientation (Euler angles)
euler_angles = np.array([...])

# Initialize calculator
calculator = SAWCalculator(material, euler_angles)

# Calculate SAW speed for a given propagation angle
angle_deg = 30
saw_speed, psaw_speed, polarization = calculator.get_saw_speed(angle_deg)

print(f"SAW speed at {angle_deg} deg: {saw_speed} m/s")
```

## Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` file (TODO: create this file if needed) for guidelines.

## License

This project is licensed under the MIT License - see the `LICENSE` file (TODO: create this file) for details.  
