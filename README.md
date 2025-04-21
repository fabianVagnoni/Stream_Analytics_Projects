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
2. Create a virtual environment and add `RIDES_PRIMARY_CONNECTION_STRING` and `SPECIAL_PRIMARY_CONNECTION_STRING` from Azure Event Hubs. 
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
- `Special Events` topic has 2 partitions for lower volume special events data.

## Troubleshooting
- **Missing Modules**: Ensure all imported custom modules are available.
- **AVRO Errors**: Install `fastavro` and verify schema files exist.
- **Memory Issues**: Reduce `--batch-size` or simulation duration for large datasets.

# Stream Analytics Application

This section of the repository contains a stream analytics application that processes ride and special events data from Azure Event Hubs.

## Architecture

The overall system architecture connects the data generator to analytics components:

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Data Generator  │───▶│ Azure Event  │───▶│ Stream Analytics│───▶│ Azure Blob      │
│ (Python)        │    │ Hubs         │    │ (Spark)         │    │ Storage         │
└─────────────────┘    └──────────────┘    └─────────────────┘    └────────┬────────┘
                                                                           │
                                                                           ▼
                                                                  ┌─────────────────┐
                                                                  │ Analytics       │
                                                                  │ Dashboard       │
                                                                  │ (Streamlit)     │
                                                                  └─────────────────┘
```

## Stream Processing Structure

The project follows a modular structure:


```
.
├── main.py                        # Data generator main script
├── generate_static_data.py        # User and driver data generation
├── geographic_model.py            # City map and location simulation
├── temporal_patterns.py           # Time-based demand patterns
├── ride_simulator.py              # Ride event simulation
├── special_events.py              # Special event generation
├── avro_serializer.py             # AVRO serialization utilities
├── modular_consumer.py            # Stream analytics main script
├── 0_Intro_Page.py                # Dashboard main entry point
├── pages/                         # Dashboard pages
│   ├── 1_Ride_Operations_Analytics.py   # Operations analysis page
│   ├── 2_Carbon_Footprint_Tracker.py    # Carbon footprint analysis page
│   └── 3_Outlier_Rides_Segmentation.py  # Customer segmentation page
├── stream_processing/             # Stream processing modules
│   ├── __init__.py
│   ├── aggregation.py             # Data aggregation functions
│   ├── config.py                  # Configuration settings
│   ├── enrichment.py              # Data enrichment functions
│   ├── schema_utils.py            # Schema handling utilities
│   ├── setup.py                   # Environment setup
│   ├── spark_session.py           # Spark session creation
│   ├── stream_readers.py          # Stream reading functions
│   ├── stream_writers.py          # Stream writing functions
│   └── storage_sender.py          # Azure storage interactions
├── utils/                         # Dashboard utility functions
│   ├── data_loader.py             # Data loading functions
│   └── visualizations.py          # Visualization components
├── schemas/                       # Schema definitions
│   ├── ride_datafeed_schema.json
│   ├── riders_drivers_avro-schemas.json
│   └── special_events_schema.json
├── advanced_analytics_models/     # Saved ML models
├── requirements.txt               # Project dependencies
├── README.md                      # This documentation file
├── README_dashboard.md            # Dashboard-specific documentation
├── Generation_plan.md             # Data generation strategy document
├── .env                           # Environment variables (not in repo)
├── advanced_analytics.ipynb       # Advanced analytics notebook
└── Basic&Intermediate_Analytics.ipynb  # Basic/intermediate analytics
```

## Stream Processing Features

The stream analytics application provides:

- **Real-time Processing**: Consumes ride and special events from Azure Event Hubs
- **AVRO Deserialization**: Properly decodes AVRO-formatted messages
- **Data Enrichment**: Adds time features and correlates ride data with special events
- **User Aggregations**: Creates user profiles and behavior vectors
- **Multi-destination Output**: Writes processed data to memory, parquet files, and Azure Blob Storage

## Usage

You can run the stream processing application using this command:

```bash
python modular_consumer.py
```

The application:
1. Connects to Azure Event Hubs
2. Reads ride events and special events
3. Deserializes AVRO messages
4. Processes and enriches the data
5. Performs aggregations for user profile creation
6. Writes results to memory, parquet files, and Azure Blob Storage

## Stream Processing Functionality

The application performs several key operations:
- Deserializes AVRO messages from Event Hubs
- Enriches ride data with time components (hour, day, month, etc.)
- Correlates rides with special events (concerts, sports, weather)
- Creates user behavioral vectors through aggregations
- Persists processed data to parquet files and Azure Blob Storage

# Analytics Dashboard

The repository also includes a multi-page Streamlit dashboard for analyzing ride-hailing operations data.

## Dashboard Features

The dashboard consists of three main pages:

1. **Ride Operations Analytics**:
   - Real-time ride operations KPIs (active rides, requested rides, completed rides)
   - Cancellation rate and request acceptance ratio
   - Customer satisfaction metrics
   - Driver categorization (Gold, Silver, Bronze)
   - Hourly ride activity analysis
   - Geographic distribution of rides

2. **Carbon Footprint Tracker**:
   - CO2 emissions metrics and visualization
   - Trees needed for carbon offset
   - Hourly and daily emissions patterns
   - Emissions analysis by traffic level and delay
   - Geographical emissions distribution
   - Vehicle type emissions comparison

3. **Outlier Rides & Customer Segmentation**:
   - Outlier detection and analysis
   - Variation clusters visualization
   - Customer segmentation
   - Segment characteristics and distribution
   - Outlier rides table and details

## Running the Dashboard

Run the Streamlit dashboard:

```bash
streamlit run 0_Intro_Page.py
```

The dashboard will be accessible at http://localhost:8501 in your web browser.

## Dashboard Organization

- `0_Intro_Page.py`: Main entry point for the dashboard
- `pages/`: Contains the three main dashboard pages
  - `1_Ride_Operations_Analytics.py`
  - `2_Carbon_Footprint_Tracker.py`
  - `3_Outlier_Rides_Segmentation.py`
- `utils/`: Utility modules
  - `data_loader.py`: Functions for loading data from Azure or local files
  - `visualizations.py`: Reusable visualization components

## Advanced Analytics

The project also includes Jupyter notebooks for more sophisticated analytics:
- `Basic&Intermediate_Analytics.ipynb`: Statistical analysis of ride patterns
- `advanced_analytics.ipynb`: Machine learning models for prediction and clustering

# Project-Wide Information

## Dependencies

See requirements.txt for the full list of dependencies, which includes:
- fastavro
- faker
- pandas
- numpy
- confluent-kafka
- pyspark
- findspark
- requests
- pyarrow
- matplotlib
- seaborn
- fastparquet
- scikit-learn
- shap
- azure-storage-blob
- azure-eventhub
- dotenv
- streamlit
- plotly
- altair

## Environment Setup

For a complete setup, create a `.env` file with:
```
RIDES_PRIMARY_CONNECTION_STRING=your_event_hub_connection_string
SPECIAL_PRIMARY_CONNECTION_STRING=your_event_hub_connection_string
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account_name
AZURE_STORAGE_ACCOUNT_KEY=your_storage_account_key
AZURE_BLOB_CONTAINER_NAME=your_container_name
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
```

## Future Improvements

- Add more comprehensive documentation
- Implement unit tests
- Add more advanced analytics components
- Real-time pricing optimization
- Predictive maintenance for vehicles
- Driver-passenger matching algorithms
- Enhanced fraud detection
- Mobile dashboard version

## License

This project is provided as-is for educational and demonstration purposes.

## Contact

For questions or contributions related to this project, please open an issue or contact the maintainers.
