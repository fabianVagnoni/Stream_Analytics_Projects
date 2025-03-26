# Ride-Hailing Data Generator

This project is a Python-based synthetic data generator for ride-hailing services. It simulates realistic ride-hailing data, including users, drivers, ride events, and special events (e.g., concerts, sports events, weather impacts, and system outages). The generated data can be output in both JSON and AVRO formats, making it suitable for testing data pipelines, analytics systems, or machine learning models.

## Features
- **Static and Dynamic Data**: Generates user and driver profiles with static (e.g., ID, name) and dynamic (e.g., ride counts, ratings) attributes.
- **Geographic Simulation**: Models a city map with zones and coordinates.
- **Temporal Patterns**: Incorporates hourly demand, traffic conditions, and holidays.
- **Special Events**: Simulates concerts, sports events, weather disruptions, system outages, and fraudulent behavior.
- **Flexible Output**: Supports JSON and AVRO formats for data serialization.
- **Configurable**: Adjustable via command-line arguments for scale, time range, and event frequency.

## Requirements
- Python 3.6+
- Required Python packages: see `requirements.txt`.
  - Custom modules (included): `generate_static_data`, `geographic_model`, `temporal_patterns`, `ride_simulator`, `special_events`, `avro_serializer`

## Installation
1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd Stream_Analytics_Projects
   ```
2. Create a virutal environment and add `RIDES_PRIMARY_CONNECTION_STRING` and `SPECIAL_PRIMARY_CONNECTION_STRING` from Azure Event Hubs. 
3. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify the custom modules (`generate_static_data.py`, etc.) are in the same directory as `main.py`.

## Usage
Run the script using the command line with optional arguments to customize the simulation.

### Basic Command
```bash
python3 main.py --output ./data
```

### Full Command with Options
```bash
python3 main.py \
  --output ./data \
  --start-date 2025-01-01 \
  --end-date 2025-01-02 \
  --drivers 100 \
  --users 300 \
  --base-demand 100 \
  --city "Madrid" \
  --num-concerts 2 \
  --num-sports 1 \
  --num-weather 1 \
  --batch-size 1000 \
  --random-seed 42 \
  --stream-to-eventhubs
```

### Command-Line Arguments
| Argument              | Type    | Default             | Description                                      |
|-----------------------|---------|---------------------|--------------------------------------------------|
| `--output`            | str     | `./data`            | Output directory for generated data             |
| `--stream-to-eventhubs`| str     | False              | Push events to Azure Events Hub                 |
| `--json-only`         | flag    | False               | Generate only JSON output (no AVRO)             |
| `--avro-only`         | flag    | False               | Generate only AVRO output (no JSON)             |
| `--start-date`        | str     | None                | Start date (YYYY-MM-DD)                         |
| `--end-date`          | str     | None                | End date (YYYY-MM-DD)                           |
| `--days`              | int     | 30                  | Number of days if no start/end dates            |
| `--users`             | int     | 1000                | Number of users to generate                     |
| `--drivers`           | int     | 500                 | Number of drivers to generate                   |
| `--base-demand`       | int     | 100                 | Base hourly ride demand                         |
| `--ride-schema`       | str     | `ride_datafeed_schema.json` | Path to ride event schema              |
| `--user-driver-schema`| str     | `riders_drivers_avro-schemas.json` | Path to user/driver schema      |
| `--city`              | str     | `San Francisco`     | City name for simulation                        |
| `--no-special-events` | flag    | False               | Disable special events                          |
| `--num-concerts`      | int     | 1                   | Number of concert events                        |
| `--num-sports`        | int     | 1                   | Number of sports events                         |
| `--num-weather`       | int     | 1                   | Number of weather events                        |
| `--batch-size`        | int     | 1000, max=1500      | Batch size for processing/saving events         |
| `--random-seed`       | int     | None                | Random seed for reproducibility                |

### Example Output
Running the script generates files in the specified `--output` directory:
- `users_static.json` / `.avro`: Static user data
- `users_dynamic.json` / `.avro`: Dynamic user data
- `drivers_static.json` / `.avro`: Static driver data
- `drivers_dynamic.json` / `.avro`: Dynamic driver data
- `<city>_map.json`: City map with zones
- `special_events.json`: Special event details (if enabled)
- `ride_events.json` / `.avro`: Ride event data
- `simulation_summary.json`: Summary statistics

## Output Formats
- **JSON**: Human-readable, suitable for debugging or small-scale use.
- **AVRO**: Binary format, efficient for big data systems (requires schema files).

### Schema Files
- `ride_datafeed_schema.json`: Defines the structure of ride events.
- `riders_drivers_avro-schemas.json`: Defines user and driver data structures.

Ensure these files are present or update the `--ride-schema` and `--user-driver-schema` arguments if using custom schemas.

## Logging
The script logs progress and errors to the console. Log levels include `INFO` for general updates and `ERROR` for issues.

## Extending the Generator
- Add new special event types in `special_events.py`.
- Modify demand or traffic patterns in `temporal_patterns.py`.
- Update schemas in the JSON/AVRO schema files.

## Event Hub Design Choices
- `Ride Events` topic has 4 partitions due to the rather large volume of events generated.
- 

## Troubleshooting
- **Missing Modules**: Ensure all imported custom modules are available.
- **AVRO Errors**: Install `fastavro` and verify schema files exist.
- **Memory Issues**: Reduce `--batch-size` or simulation duration for large datasets.

## License
This project is unlicensed and provided as-is for educational or testing purposes.

## Contact
For questions or contributions, please open an issue or contact the maintainers.
