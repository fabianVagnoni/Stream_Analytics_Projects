# Ride Analytics Dashboard

A multi-page Streamlit dashboard for analyzing ride-hailing operations data. This dashboard provides insights into ride operations, carbon footprint tracking, and customer segmentation.

## Features

The dashboard consists of three main pages:

1. **Ride Operations and Customer Analytics**:
   - Key KPIs for ride operations (active rides, requested rides, completed rides)
   - Cancellation rate and request acceptance ratio
   - Customer satisfaction metrics
   - Driver categorization (Gold, Silver, Bronze)
   - Hourly ride activity analysis
   - Real-time ride efficiency index

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

## Prerequisites

- Python 3.8+
- Required Python packages listed in `requirements.txt`
- Access to Azure Blob Storage with ride data (or local data files)

## Installation and Setup

1. Clone the repository:
   ```
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Azure Blob Storage credentials in a `.env` file:
   ```
   AZURE_STORAGE_ACCOUNT_NAME=your_account_name
   AZURE_STORAGE_ACCOUNT_KEY=your_account_key
   AZURE_BLOB_CONTAINER_NAME=your_container_name
   ```

## Running the Dashboard

Run the Streamlit application:
```
streamlit run dashboard.py
```

The dashboard will be accessible at http://localhost:8501 in your web browser.

## Data Structure

The dashboard expects the following data in either Azure Blob Storage or local files:

- `rides/ride_events.json`: Events data for ride operations
- `drivers_dynamic.json`: Dynamic driver data (ratings, rides count, etc.)
- `drivers_static.json`: Static driver information (vehicle type, etc.)
- `special_events.json`: Special events data
- `user_vectors/users_static.json`: Static user information

## Dashboard Organization

- `dashboard.py`: Main entry point for the dashboard
- `pages/`: Contains the three main dashboard pages
- `utils/`: Utility modules
  - `data_loader.py`: Functions for loading data from Azure or local files
  - `data_processing.py`: Data processing and transformation functions
  - `visualizations.py`: Reusable visualization components

## Future Enhancements

- Real-time data streaming integration
- Predictive analytics for demand forecasting
- Driver performance tracking and optimization
- Advanced customer segmentation with ML
- Mobile responsiveness improvements

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 