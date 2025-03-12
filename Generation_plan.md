# Ride-Hailing Data Generation Strategy

## Overview

This document outlines a comprehensive strategy for generating realistic, correlated synthetic data for a ride-hailing application, based on the provided AVRO schemas. The goal is to create data that exhibits realistic patterns and contains sufficient anomalies and edge cases to support meaningful analytics in future milestones.

## Core Approach: Hybrid Simulation Framework

We recommend a hybrid approach combining multiple data generation techniques:

1. **Probabilistic modeling** for temporal and geographic patterns
2. **Agent-based simulation** for driver and user behavior 
3. **Template-based generation** for common ride scenarios
4. **Manually crafted edge cases** for testing anomaly detection

## Key Components

### 1. Persona-Based Static Data Generation

For users and drivers, create distinct profiles with specific characteristics:

#### Driver Personas
- **Full-time professionals (30%)**: High experience, low cancellation rates, high ratings
- **Evening part-timers (25%)**: Medium experience, medium cancellation rates, good ratings
- **Weekend warriors (20%)**: Lower experience, higher cancellation rates, varied ratings
- **Newbies (15%)**: Very low experience, high cancellation rates, lower/varied ratings
- **Inconsistent drivers (10%)**: Varied experience, very high cancellation rates, unpredictable ratings

#### User Personas
- **Daily commuters (40%)**: Predictable patterns, low cancellations, consistent ratings
- **Business travelers (15%)**: Airport rides, higher fares, good ratings
- **Weekend socialites (25%)**: Night/evening rides, varied ratings, higher cancellations
- **Occasional users (20%)**: Infrequent usage, lower predictability, varied ratings

Each persona will have different behavior affecting ride frequency, timing, vehicle preferences, cancellation tendencies, and rating behaviors.

### 2. Time-Aware Simulation

Create a simulation engine incorporating realistic temporal patterns:

- **Hourly demand curves**:
  - Weekday morning peak (7-9am)
  - Weekday evening peak (5-7pm)
  - Weekend evening/night peaks (7-11pm)
  - Low demand periods (2-5am)

- **Weekly patterns**:
  - Monday (90% of average demand)
  - Tuesday-Wednesday (100% of average demand)
  - Thursday (110% of average demand)
  - Friday (130% of average demand)
  - Saturday (110% of average demand)
  - Sunday (90% of average demand)

- **Monthly/Seasonal variations**:
  - Weather impacts
  - Holiday effects
  - Special events

### 3. Geographic Distribution Model

Create a virtual city map with distinct zones:

- **Downtown business district**: High morning arrivals, high evening departures
- **Residential neighborhoods**: Morning departures, evening arrivals
- **Airport**: Consistent demand with flight schedule peaks
- **Entertainment districts**: Evening/night demand peaks
- **Shopping centers**: Daytime and weekend peaks

Implement realistic travel patterns between these areas based on time of day and user type.

### 4. Ride Lifecycle State Machine

Model complete ride flow with realistic transitions:

```
RIDE_REQUESTED → 
  ├─ [Success] → DRIVER_ASSIGNED →
  │                ├─ [Success] → DRIVER_ARRIVED → RIDE_STARTED → RIDE_COMPLETED → PAYMENT_COMPLETED → 
  │                │                                                                  ├─ USER_RATED_DRIVER
  │                │                                                                  └─ DRIVER_RATED_USER
  │                └─ [Failure] → RIDE_CANCELED_BY_DRIVER
  └─ [Failure] → RIDE_CANCELED_BY_USER
```

Critical factors influencing transitions:
- Distance between driver and pickup
- Time of day (affects cancellation rates)
- User/driver historical cancellation rates
- Surge pricing (affects user cancellation)
- Traffic conditions

### 5. Special Events and Anomaly Injection

Create specific patterns for meaningful analytics:

- **High-demand events**: 
  - Concerts or sports games with extreme surge pricing
  - 300-500 rides concentrated in time and location

- **Severe weather events**:
  - Extended ride durations (20-50% longer)
  - Higher cancellation rates
  - Limited driver availability

- **System disruptions**:
  - Brief service outages (1-3 hours)
  - High request failures
  - Ride cancellations

- **Fraudulent patterns**:
  - Suspicious circular rides
  - Unusual short/long rides
  - Rating manipulation
  - 5-10 "fraudulent" users with distinct patterns

- **Driver and user outliers**:
  - Super-drivers with 3-5x average rides
  - Riders with extreme frequency or spending

### 6. Data Correlation Implementation

Critical correlations to include:

- **User-Driver matching**: Drivers with higher ratings get more ride requests
- **Time-Geography correlation**: Rush hour affects different city areas differently
- **Surge pricing effects**: Higher surge correlates with more cancellations
- **Driver experience impact**: More experienced drivers have shorter pickup times
- **Traffic impact**: Rush hours have longer ride durations for the same route
- **Rating correlations**: Longer wait times and traffic affect ratings negatively
- **Payment-rating correlation**: Higher fares tend to yield more critical ratings

## Implementation Workflow

1. Generate static user and driver data with defined personas
2. Create the geographic and temporal demand models
3. Implement the ride simulation engine with the state machine
4. Generate baseline normal ride data
5. Inject special events and anomalies
6. Validate data patterns and correlations
7. Export to JSON and AVRO formats

## Key Libraries and Tools

- **Faker**: For realistic user/driver details
- **NumPy/SciPy**: For statistical distributions and random sampling
- **Pandas**: For data manipulation and validation
- **FastAVRO**: For AVRO serialization
- **GeoPy**: For geographical distance calculations
- **Matplotlib/Seaborn**: For validating data distributions

## Expected Outputs

- 500-1,000 users with static and dynamic data
- 200-500 drivers with static and dynamic data
- 10,000-50,000 ride events covering a 3-month period
- 5-10 special event periods with distinct patterns
- JSON and AVRO formatted data files

## Validation Strategy

To ensure data quality and realism:
- Verify temporal patterns match expected distributions
- Confirm geographic hotspots align with zone definitions
- Check correlation between related metrics (e.g., rating vs. wait time)
- Validate that injected anomalies are detectable but not obvious
- Ensure all data follows schema definitions
