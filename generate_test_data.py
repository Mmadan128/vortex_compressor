"""
Generate test binary datasets for compression benchmarking.

Creates various types of structured binary data that simulate:
- High-energy detector event data (ATLAS-like)
- Time-series sensor telemetry
- Network packet captures
- Industrial control system logs
"""

import numpy as np
import struct
import argparse
from pathlib import Path
from typing import Optional
import json


def generate_atlas_style_events(num_events: int = 10000, output_file: str = "atlas_events.bin"):
    """
    Generates ATLAS-like detector event data with realistic structure.
    
    Each event contains:
    - Event header (run number, event number, timestamp, trigger bits)
    - Track data (momentum, charge, detector hits)
    - Calorimeter energy deposits
    - Muon system hits
    
    This simulates the binary format of high-energy detector readout.
    """
    print(f"Generating {num_events} ATLAS-style detector events...")
    
    with open(output_file, "wb") as f:
        for event_num in range(num_events):
            run_number = 450000 + (event_num // 1000)
            event_number = event_num
            timestamp = 1704067200 + event_num * 25  # 25ns bunch crossing
            trigger_bits = np.random.choice([0x01, 0x02, 0x04, 0x08, 0x10], 
                                           p=[0.4, 0.3, 0.15, 0.1, 0.05])
            
            header = struct.pack('<IIIQ', run_number, event_number, trigger_bits, timestamp)
            f.write(header)
            
            num_tracks = np.random.poisson(15)
            f.write(struct.pack('<H', num_tracks))
            
            for _ in range(num_tracks):
                pt = np.abs(np.random.normal(45.0, 20.0))
                eta = np.random.normal(0.0, 1.5)
                phi = np.random.uniform(-np.pi, np.pi)
                charge = np.random.choice([-1, 1])
                nhits = np.random.randint(8, 32)
                
                track_data = struct.pack('<fffbB', pt, eta, phi, charge, nhits)
                f.write(track_data)
            
            num_calo_clusters = np.random.poisson(25)
            f.write(struct.pack('<H', num_calo_clusters))
            
            for _ in range(num_calo_clusters):
                energy = np.abs(np.random.exponential(5.0))
                eta = np.random.uniform(-2.5, 2.5)
                phi = np.random.uniform(-np.pi, np.pi)
                layer = np.random.randint(0, 4)
                
                calo_data = struct.pack('<fffB', energy, eta, phi, layer)
                f.write(calo_data)
            
            num_muons = np.random.poisson(2)
            f.write(struct.pack('<B', num_muons))
            
            for _ in range(num_muons):
                pt = np.abs(np.random.normal(30.0, 15.0))
                eta = np.random.normal(0.0, 1.2)
                phi = np.random.uniform(-np.pi, np.pi)
                quality = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
                
                muon_data = struct.pack('<fffB', pt, eta, phi, quality)
                f.write(muon_data)
    
    file_size = Path(output_file).stat().st_size
    print(f"Created {output_file}: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    return output_file


def generate_time_series_sensors(duration_seconds: int = 3600, 
                                 num_sensors: int = 50,
                                 sampling_rate_hz: int = 100,
                                 output_file: str = "sensor_telemetry.bin"):
    """
    Generates time-series sensor data with periodic patterns and noise.
    
    Simulates industrial sensor arrays with:
    - Temperature sensors (slow drift + daily cycles)
    - Pressure sensors (fast fluctuations)
    - Flow rate sensors (step changes)
    - Vibration sensors (high-frequency oscillations)
    """
    print(f"Generating {duration_seconds}s of {num_sensors} sensor time series...")
    
    num_samples = duration_seconds * sampling_rate_hz
    
    with open(output_file, "wb") as f:
        header = struct.pack('<HHI', num_sensors, sampling_rate_hz, num_samples)
        f.write(header)
        
        for sample_idx in range(num_samples):
            timestamp_ms = sample_idx * (1000 // sampling_rate_hz)
            f.write(struct.pack('<Q', timestamp_ms))
            
            for sensor_id in range(num_sensors):
                sensor_type = sensor_id % 4
                
                if sensor_type == 0:
                    base_value = 20.0 + 5.0 * np.sin(2 * np.pi * sample_idx / (3600 * sampling_rate_hz))
                    noise = np.random.normal(0, 0.5)
                    value = base_value + noise
                    
                elif sensor_type == 1:
                    base_value = 101325.0
                    oscillation = 100.0 * np.sin(2 * np.pi * sample_idx / (10 * sampling_rate_hz))
                    noise = np.random.normal(0, 20.0)
                    value = base_value + oscillation + noise
                    
                elif sensor_type == 2:
                    step = int(sample_idx / (300 * sampling_rate_hz))
                    base_value = 50.0 + (step % 5) * 10.0
                    noise = np.random.normal(0, 2.0)
                    value = base_value + noise
                    
                else:
                    freq = 50.0
                    value = 0.5 * np.sin(2 * np.pi * freq * sample_idx / sampling_rate_hz)
                    value += np.random.normal(0, 0.1)
                
                f.write(struct.pack('<f', value))
    
    file_size = Path(output_file).stat().st_size
    print(f"Created {output_file}: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    return output_file


def generate_network_packets(num_packets: int = 100000,
                            output_file: str = "network_capture.bin"):
    """
    Generates network packet capture data with realistic protocol patterns.
    
    Simulates TCP/IP packets with:
    - Ethernet headers
    - IP headers with common addresses
    - TCP/UDP payloads
    - Protocol-specific patterns (HTTP, DNS, etc.)
    """
    print(f"Generating {num_packets} network packet captures...")
    
    common_ips = [
        [192, 168, 1, i] for i in range(1, 20)
    ] + [
        [10, 0, 0, i] for i in range(1, 10)
    ]
    
    common_ports = [80, 443, 22, 53, 8080, 3306, 5432, 6379]
    
    with open(output_file, "wb") as f:
        for _ in range(num_packets):
            packet_size = np.random.choice([64, 128, 256, 512, 1024, 1500], 
                                          p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
            
            src_ip = common_ips[np.random.randint(0, len(common_ips))]
            dst_ip = common_ips[np.random.randint(0, len(common_ips))]
            src_port = np.random.choice(common_ports)
            dst_port = np.random.choice(common_ports)
            
            protocol = np.random.choice([6, 17], p=[0.8, 0.2])  # TCP or UDP
            
            header = struct.pack('<HBBBBBBBBHH',
                                packet_size,
                                src_ip[0], src_ip[1], src_ip[2], src_ip[3],
                                dst_ip[0], dst_ip[1], dst_ip[2], dst_ip[3],
                                protocol,
                                src_port)
            f.write(header)
            
            payload_size = packet_size - len(header)
            if dst_port == 80 or dst_port == 8080:
                payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
                payload = payload[:payload_size].ljust(payload_size, b'\x00')
            elif dst_port == 443:
                payload = b"\x16\x03\x01" + np.random.bytes(payload_size - 3)
            else:
                payload = np.random.bytes(payload_size)
            
            f.write(payload)
    
    file_size = Path(output_file).stat().st_size
    print(f"Created {output_file}: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    return output_file


def generate_structured_log(num_entries: int = 50000,
                           output_file: str = "system_logs.bin"):
    """
    Generates structured binary log data with repeating patterns.
    
    Simulates system logs with:
    - Timestamps
    - Log levels
    - Component IDs
    - Error codes
    - Message payloads
    """
    print(f"Generating {num_entries} structured log entries...")
    
    log_levels = [1, 2, 3, 4, 5]  # DEBUG, INFO, WARN, ERROR, CRITICAL
    level_probs = [0.1, 0.6, 0.2, 0.08, 0.02]
    
    components = list(range(20))
    
    error_codes = [0, 100, 200, 404, 500, 503]
    
    with open(output_file, "wb") as f:
        base_timestamp = 1704067200
        
        for entry_idx in range(num_entries):
            timestamp = base_timestamp + entry_idx
            log_level = np.random.choice(log_levels, p=level_probs)
            component_id = np.random.choice(components)
            
            if log_level >= 4:
                error_code = np.random.choice([404, 500, 503])
            else:
                error_code = 0
            
            thread_id = np.random.randint(1, 32)
            
            header = struct.pack('<QBBHH', timestamp, log_level, component_id, 
                               error_code, thread_id)
            f.write(header)
            
            msg_templates = [
                b"Connection established to remote host",
                b"Request processed successfully",
                b"Cache hit for key",
                b"Database query executed",
                b"Configuration reload initiated",
                b"Authentication failed for user",
                b"Timeout waiting for response",
                b"Resource allocation completed"
            ]
            
            msg = msg_templates[entry_idx % len(msg_templates)]
            msg_len = len(msg)
            f.write(struct.pack('<H', msg_len))
            f.write(msg)
    
    file_size = Path(output_file).stat().st_size
    print(f"Created {output_file}: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    return output_file


def generate_all_test_datasets(output_dir: str = "data"):
    """Generates all test dataset types."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Generating Test Datasets for Vortex-Codec")
    print("=" * 70)
    
    datasets = {}
    
    print("\n[1/4] High-energy detector events (ATLAS-style)...")
    datasets['atlas'] = generate_atlas_style_events(
        num_events=10000,
        output_file=str(output_path / "atlas_events.bin")
    )
    
    print("\n[2/4] Time-series sensor telemetry...")
    datasets['sensors'] = generate_time_series_sensors(
        duration_seconds=1800,
        num_sensors=50,
        sampling_rate_hz=100,
        output_file=str(output_path / "sensor_telemetry.bin")
    )
    
    print("\n[3/4] Network packet captures...")
    datasets['network'] = generate_network_packets(
        num_packets=50000,
        output_file=str(output_path / "network_capture.bin")
    )
    
    print("\n[4/4] Structured system logs...")
    datasets['logs'] = generate_structured_log(
        num_entries=50000,
        output_file=str(output_path / "system_logs.bin")
    )
    
    print("\n" + "=" * 70)
    print("Dataset Generation Complete")
    print("=" * 70)
    
    manifest = {
        'datasets': {
            'atlas_events': {
                'path': datasets['atlas'],
                'description': 'High-energy detector event data',
                'size_bytes': Path(datasets['atlas']).stat().st_size,
                'type': 'structured_events'
            },
            'sensor_telemetry': {
                'path': datasets['sensors'],
                'description': 'Time-series sensor measurements',
                'size_bytes': Path(datasets['sensors']).stat().st_size,
                'type': 'time_series'
            },
            'network_capture': {
                'path': datasets['network'],
                'description': 'Network packet capture data',
                'size_bytes': Path(datasets['network']).stat().st_size,
                'type': 'network_traffic'
            },
            'system_logs': {
                'path': datasets['logs'],
                'description': 'Binary structured log entries',
                'size_bytes': Path(datasets['logs']).stat().st_size,
                'type': 'logs'
            }
        }
    }
    
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest written to: {manifest_file}")
    print("\nTo train on these datasets:")
    print("  python train_example.py --data data/atlas_events.bin")
    print("  python train_example.py --data data/sensor_telemetry.bin")
    
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test datasets for compression")
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for generated datasets')
    parser.add_argument('--atlas-events', type=int, default=10000,
                       help='Number of ATLAS-style events to generate')
    parser.add_argument('--sensor-duration', type=int, default=1800,
                       help='Duration of sensor data in seconds')
    parser.add_argument('--network-packets', type=int, default=50000,
                       help='Number of network packets to generate')
    parser.add_argument('--log-entries', type=int, default=50000,
                       help='Number of log entries to generate')
    
    args = parser.parse_args()
    
    generate_all_test_datasets(output_dir=args.output_dir)
