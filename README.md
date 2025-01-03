# pqopen-lib

`pqopen-lib` is a Python library designed for advanced power system and power quality analysis. It provides tools for creating and analyzing power systems, detecting events, managing data storage, and more. Built with modularity and flexibility in mind, it supports single-phase to multi-phase systems and complies with IEC standards for power quality analysis.



## Features

- **Power System Modeling**: Create single-phase or multi-phase power systems with detailed phase configuration.
- **Power Quality Analysis**: Perform harmonic, fluctuation, and power quality calculations in compliance with standards like IEC 61000-4-7 and IEC 61000-4-30.
- **Event Detection**: Detect and classify events such as overvoltage, undervoltage, and other anomalies in real-time.
- **Data Storage and Export**: Manage time-series and aggregated data with support for various storage backends like CSV, and MQTT
- **Zero-Crossing Detection**: Enable accurate cycle-by-cycle processing for fundamental frequency synchronization.
- **High Performance**: Efficiently handle high-resolution waveform data and sampling rates.



## Installation

Ensure you have Python 3.11 or later installed. To install `pqopen-lib` along with its dependencies:

```bash
pip install pqopen-lib
```



## Documentation

Detailed [documentation](https://docs.daqopen.com/reference/pqopen/) is available for each module and function. Key modules include:

- **`powersystem`**: Define and manage power systems and phases.
- **`powerquality`**: Perform power quality and harmonic analyses.
- **`eventdetector`**: Detect and analyze power events.
- **`storagecontroller`**: Manage and export time-series and aggregated data.
- **`zcd`**: Handle zero-crossing detection for waveform synchronization.



## Use Cases

- **Education**: Learn to understand how power and power quality analysis works
- **Power Quality Monitoring**: Ensure compliance with power quality standards.
- **Industrial Power Systems**: Monitor and analyze complex, multi-phase systems.
- **Research and Development**: Use as a reference platform for power quality testing



## Getting Started

Here’s a quick example to get you started:

### Create a Simple Power System

```python
import numpy as np

from pqopen.powersystem import PowerSystem
from daqopen.channelbuffer import AcqBuffer

samplerate = 10_000 # Hz
signal_duration = 1.0 # seconds

voltage_magnitude = 230.0 # Volt
current_magnitude = 10.0 # Ampere

frequency = 50.0 # Hz

# Create time signal
t = np.linspace(0,int(signal_duration),int(samplerate*signal_duration),endpoint=False)
# Create voltage signal
u = voltage_magnitude*np.sqrt(2)*np.sin(2*np.pi*t*frequency)
# Create current signal
i = current_magnitude*np.sqrt(2)*np.sin(2*np.pi*t*frequency)

# Create Channel/Buffer for input waveform
ch_t = AcqBuffer()
ch_u = AcqBuffer()
ch_i = AcqBuffer()

# Create minimal power system
my_power_system = PowerSystem(zcd_channel=ch_u,
                              input_samplerate=samplerate)

# Create power phase and append to power system
my_power_system.add_phase(u_channel=ch_u, i_channel=ch_i)

# Add data to channels (we can apply all data at once because the 
# buffer is big enough to hold the test dataset)
ch_t.put_data(t)
ch_u.put_data(u)
ch_i.put_data(i)

# Perform calculation
my_power_system.process()

# View the results
for ch_name, ch_buffer in my_power_system.output_channels.items():
    print(f"{ch_name:<14} {ch_buffer.last_sample_value:.2f} {ch_buffer.unit}")
```



## Contributing

Contributions are welcome! Please open issues for bugs or feature requests and submit pull requests for improvements.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m 'Add my feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a pull request.



## License

This project is licensed under the MIT License. See the LICENSE file for details.

------

For any questions or support, feel free to reach out via the issues page or contact me michael@daqopen.com