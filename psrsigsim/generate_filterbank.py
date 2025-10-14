#!/usr/bin/env python3
"""
Pulsar Filterbank Simulation with Effelsberg Telescope
Generates realistic filterbank data with noise and dispersion effects
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import psrsigsim, if not available we'll use simplified simulation
try:
    import psrsigsim as pss
    import astropy.units as u
    from astropy.time import Time
    HAS_PSRSIGSIM = True
    print("psrsigsim available - using full simulation")
except ImportError:
    HAS_PSRSIGSIM = False
    print("psrsigsim not available - using simplified simulation")

class EfelsbergTelescope:
    """Simplified Effelsberg telescope class for simulation"""
    def __init__(self):
        self.name = "Effelsberg"
        self.latitude = 50.525    # degrees North
        self.longitude = 6.883    # degrees East  
        self.elevation = 369.0    # meters above sea level
        self.diameter = 100.0     # 100m dish diameter
        self.area = 7854.0        # effective area in m^2 (π * 50^2)
        self.Tsys = 30.0          # system temperature in K (typical L-band)
        self.efficiency = 0.7     # aperture efficiency

class SimplePulsar:
    """Simplified pulsar class for simulation"""
    def __init__(self, period, dm, flux, name):
        self.period = period      # seconds
        self.dm = dm             # pc cm^-3
        self.Sm = flux           # mJy
        self.name = name

def create_effelsberg_telescope():
    """Create Effelsberg telescope object"""
    if HAS_PSRSIGSIM:
        telescope = pss.telescope.Telescope(
            name="Effelsberg",
            latitude=50.525,
            longitude=6.883,
            elevation=369.0,
            diameter=100.0,
            area=7854.0,
            Tsys=30.0,
            efficiency=0.7
        )
    else:
        telescope = EfelsbergTelescope()
    
    return telescope

def create_observation_parameters():
    """Set up realistic observation parameters for pulsar observations"""
    f_low = 1100.0     # MHz - lowest frequency
    f_high = 1700.0    # MHz - highest frequency
    nchan = 256        # number of frequency channels
    tobs = 60.0        # seconds - total observation time
    dt = 50e-6         # seconds - sampling time (50 microseconds)
    
    # Calculate derived parameters
    bandwidth = f_high - f_low
    df = bandwidth / nchan
    f_center = (f_high + f_low) / 2
    nsamp = int(tobs / dt)
    
    return {
        'f_low': f_low, 'f_high': f_high, 'nchan': nchan,
        'tobs': tobs, 'dt': dt, 'bandwidth': bandwidth,
        'df': df, 'f_center': f_center, 'nsamp': nsamp
    }

def create_pulsar():
    """Create a pulsar with realistic parameters similar to known pulsars"""
    if HAS_PSRSIGSIM:
        pulsar = pss.pulsar.Pulsar(
            period=33.085e-3,           # seconds (Crab-like)
            Sm=1000.0,                  # mJy
            profiles=[1.0],
            name="SimPulsar_J0000+0000"
        )
        pulsar.dm = 56.8  # pc cm^-3
    else:
        pulsar = SimplePulsar(
            period=33.085e-3,           # seconds (Crab-like)
            dm=56.8,                    # pc cm^-3 (similar to Crab)
            flux=1000.0,                # mJy
            name="SimPulsar_J0000+0000"
        )
    
    return pulsar

def generate_filterbank_data(pulsar, telescope, obs_params):
    """Generate realistic filterbank data with pulsar signals, dispersion, and noise"""
    print("Generating filterbank data...")
    
    # Create frequency and time arrays
    frequencies = np.linspace(obs_params['f_low'], obs_params['f_high'], obs_params['nchan'])
    times = np.arange(obs_params['nsamp']) * obs_params['dt']
    
    # Initialize data array (frequency x time)
    data = np.zeros((obs_params['nchan'], obs_params['nsamp']), dtype=np.float32)
    
    # Generate pulsar signal with dispersion
    print("Adding pulsar signal with dispersion...")
    
    # Dispersion constant: 4.148808 × 10^3 MHz^2 pc^-1 cm^3 s
    dispersion_constant = 4.148808e3
    
    # Calculate dispersion delays for each frequency
    f_ref = frequencies[-1]  # highest frequency as reference
    delays = dispersion_constant * pulsar.dm * (1/frequencies**2 - 1/f_ref**2)
    
    # Generate pulse profile (simple Gaussian)
    pulse_width = pulsar.period * 0.05  # 5% of period
    
    for i, freq in enumerate(frequencies):
        # Apply dispersion delay
        delay = delays[i]
        shifted_times = (times - delay) % pulsar.period
        
        # Create pulse profile
        pulse_profile = np.exp(-0.5 * ((shifted_times - pulsar.period/2) / pulse_width)**2)
        
        # Scale by flux and frequency dependence (typical spectral index -2)
        flux_scale = pulsar.Sm * (freq / obs_params['f_center'])**(-2.0)
        
        # Add to data
        data[i, :] = flux_scale * pulse_profile
    
    return data, frequencies, times, delays

def add_telescope_noise(data, telescope, obs_params):
    """Add realistic telescope noise based on system temperature"""
    print("Adding telescope noise...")
    
    # Calculate noise level based on radiometer equation
    bandwidth_hz = obs_params['bandwidth'] * 1e6  # MHz to Hz
    integration_time = obs_params['dt']           # seconds per sample
    channel_bandwidth = bandwidth_hz / obs_params['nchan']
    
    # Noise RMS per channel per sample
    noise_rms = telescope.Tsys / np.sqrt(channel_bandwidth * integration_time)
    
    # Convert from temperature to flux units (simplified)
    temp_to_flux = 1.0  # K to mJy conversion factor (simplified)
    noise_rms_flux = noise_rms * temp_to_flux
    
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_rms_flux, data.shape).astype(np.float32)
    
    # Add noise to signal
    noisy_data = data + noise
    
    return noisy_data, noise

def plot_filterbank_analysis(data, frequencies, times, delays, pulsar, telescope, obs_params):
    """Create comprehensive plots of the generated filterbank data"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Simulated Filterbank Data - Effelsberg Telescope', fontsize=16, fontweight='bold')
    
    # 1. Waterfall plot (frequency vs time) - show first 5 seconds
    ax1 = axes[0, 0]
    time_end_idx = min(int(5.0 / obs_params['dt']), len(times))  # first 5 seconds
    time_subset = slice(0, time_end_idx)
    
    im1 = ax1.imshow(data[:, time_subset], 
                     aspect='auto', 
                     origin='lower',
                     extent=[0, times[time_end_idx-1], frequencies[0], frequencies[-1]],
                     cmap='viridis',
                     interpolation='nearest')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (MHz)')
    ax1.set_title('Waterfall Plot (first 5 seconds)')
    plt.colorbar(im1, ax=ax1, label='Flux (mJy)')
    
    # 2. Integrated pulse profile (folded by period)
    ax2 = axes[0, 1]
    period_samples = int(pulsar.period / obs_params['dt'])
    if period_samples > 10:  # Only fold if we have enough samples per period
        n_periods = len(times) // period_samples
        folded_data = data.sum(axis=0)[:n_periods * period_samples].reshape(n_periods, period_samples)
        avg_profile = folded_data.mean(axis=0)
        phase = np.linspace(0, 1, len(avg_profile))
        ax2.plot(phase, avg_profile, 'b-', linewidth=2)
        ax2.set_xlabel('Pulse Phase')
        ax2.set_ylabel('Integrated Flux (mJy)')
        ax2.set_title(f'Average Pulse Profile (P = {pulsar.period*1000:.1f} ms)')
    else:
        # If period is too short, just show time series
        profile = np.sum(data, axis=0)
        ax2.plot(times[time_subset], profile[time_subset], 'b-', linewidth=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Integrated Flux (mJy)')
        ax2.set_title('Time Series (Integrated over Frequency)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Frequency spectrum (sum over time)
    ax3 = axes[1, 0]
    spectrum = np.sum(data, axis=1)
    ax3.plot(frequencies, spectrum, 'r-', linewidth=2)
    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('Integrated Flux (mJy⋅s)')
    ax3.set_title('Frequency Spectrum (Integrated over Time)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Dispersion delay curve
    ax4 = axes[1, 1]
    ax4.plot(frequencies, delays * 1000, 'g-', linewidth=3, label=f'DM = {pulsar.dm} pc cm⁻³')
    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('Delay (ms)')
    ax4.set_title('Dispersion Delay Curve')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('filterbank_analysis.png', dpi=150, bbox_inches='tight')
    print("Analysis plot saved to: filterbank_analysis.png")
    
    return fig

def print_metadata(data, frequencies, times, delays, pulsar, telescope, obs_params):
    """Print comprehensive metadata"""
    print("\\n" + "="*70)
    print("FILTERBANK METADATA")
    print("="*70)
    print(f"Telescope: {telescope.name}")
    print(f"  Location: {telescope.latitude:.3f}°N, {telescope.longitude:.3f}°E")
    print(f"  Diameter: {telescope.diameter} m")
    print(f"  System Temperature: {telescope.Tsys} K")
    print(f"  Efficiency: {telescope.efficiency}")
    print(f"\\nPulsar: {pulsar.name}")
    print(f"  Period: {pulsar.period*1000:.3f} ms")
    print(f"  DM: {pulsar.dm} pc cm⁻³")
    print(f"  Mean Flux: {pulsar.Sm} mJy")
    print(f"\\nObservation Parameters:")
    print(f"  Frequency Range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} MHz")
    print(f"  Center Frequency: {obs_params['f_center']:.1f} MHz")
    print(f"  Bandwidth: {obs_params['bandwidth']:.1f} MHz")
    print(f"  Channels: {obs_params['nchan']}")
    print(f"  Channel Width: {obs_params['df']:.3f} MHz")
    print(f"  Observation Time: {obs_params['tobs']:.1f} s")
    print(f"  Time Resolution: {obs_params['dt']*1e6:.1f} μs")
    print(f"  Samples: {obs_params['nsamp']:,}")
    print(f"\\nData Properties:")
    print(f"  Data Shape: {data.shape} (freq × time)")
    print(f"  Data Size: {data.nbytes/1024/1024:.2f} MB")
    print(f"  Signal Peak: {np.max(data):.3f} mJy")
    print(f"  Signal Mean: {np.mean(data):.3f} mJy")
    print(f"  Signal RMS: {np.std(data):.3f} mJy")
    print(f"\\nDispersion Properties:")
    print(f"  Max Delay: {np.max(delays)*1000:.2f} ms")
    print(f"  Min Delay: {np.min(delays)*1000:.2f} ms")
    print(f"  Delay Span: {(np.max(delays) - np.min(delays))*1000:.2f} ms")
    print(f"  Delay/Period Ratio: {np.max(delays)/pulsar.period:.2f}")

def main():
    """Main function to generate filterbank data"""
    print("Pulsar Filterbank Simulation with Effelsberg Telescope")
    print("=" * 60)
    
    # Create telescope and pulsar objects
    telescope = create_effelsberg_telescope()
    pulsar = create_pulsar()
    obs_params = create_observation_parameters()
    
    print(f"\\nTelescope: {telescope.name} ({telescope.diameter}m)")
    print(f"Pulsar: {pulsar.name} (P={pulsar.period*1000:.1f}ms, DM={pulsar.dm})")
    print(f"Observation: {obs_params['f_low']}-{obs_params['f_high']} MHz, {obs_params['tobs']}s")
    
    # Generate data
    data, frequencies, times, delays = generate_filterbank_data(pulsar, telescope, obs_params)
    
    # Add noise
    noisy_data, noise = add_telescope_noise(data, telescope, obs_params)
    
    # Save data
    print("\\nSaving data...")
    np.save('simulated_filterbank.npy', noisy_data)
    metadata = {
        'frequencies': frequencies,
        'times': times,
        'delays': delays,
        'telescope': telescope.name,
        'pulsar_name': pulsar.name,
        'dm': pulsar.dm,
        'period': pulsar.period,
        'obs_params': obs_params
    }
    np.save('filterbank_metadata.npy', metadata)
    print("Data saved to: simulated_filterbank.npy")
    print("Metadata saved to: filterbank_metadata.npy")
    
    # Create plots
    print("\\nCreating analysis plots...")
    fig = plot_filterbank_analysis(noisy_data, frequencies, times, delays, 
                                 pulsar, telescope, obs_params)
    
    # Print metadata
    print_metadata(noisy_data, frequencies, times, delays, pulsar, telescope, obs_params)
    
    print("\\n" + "="*60)
    print("Simulation completed successfully!")
    print("Generated files:")
    print("  - simulated_filterbank.npy (data)")
    print("  - filterbank_metadata.npy (metadata)")
    print("  - filterbank_analysis.png (plots)")

if __name__ == "__main__":
    main()