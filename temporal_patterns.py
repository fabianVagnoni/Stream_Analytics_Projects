"""
Temporal Patterns for Ride-Hailing Simulation

This module provides functions to model time-based patterns for ride demand,
traffic conditions, and other temporal variables affecting ride-hailing services.

Usage:
    from temporal_patterns import DemandModel
    demand_model = DemandModel()
    multiplier = demand_model.get_demand_multiplier(datetime.now())
"""

import numpy as np
from datetime import datetime, timedelta, time
import random
import calendar
import json
import os

class DemandModel:
    """
    Models time-based demand patterns for ride-hailing services.
    """
    
    def __init__(self, base_demand=100, holidays=None):
        """
        Initialize the demand model.
        
        Args:
            base_demand: Base number of ride requests per hour under normal conditions
            holidays: List of holiday dates (datetime objects) for special handling
        """
        self.base_demand = base_demand
        self.holidays = holidays or []
        
        # Predefined patterns
        self.hourly_patterns = self._create_hourly_patterns()
        self.day_of_week_factors = self._create_daily_factors()
        self.monthly_factors = self._create_monthly_factors()
        
        # Special events will be stored here
        self.special_events = []
    
    def _create_hourly_patterns(self):
        """Create hourly demand patterns for different day types"""
        # Weekday pattern has strong commuter peaks
        weekday_pattern = [
            0.2,  # 12am - very low
            0.1,  # 1am - lowest
            0.1,  # 2am - lowest
            0.1,  # 3am - lowest
            0.2,  # 4am - very low
            0.5,  # 5am - early commuters
            1.0,  # 6am - commute building
            1.8,  # 7am - morning peak
            2.2,  # 8am - highest peak
            1.7,  # 9am - commute trailing
            1.0,  # 10am - mid-morning
            1.0,  # 11am - mid-day
            1.3,  # 12pm - lunch rush
            1.2,  # 1pm - post-lunch
            1.0,  # 2pm - mid-afternoon
            0.9,  # 3pm - mid-afternoon
            1.2,  # 4pm - early commute
            1.8,  # 5pm - evening commute starts
            2.0,  # 6pm - evening peak
            1.6,  # 7pm - dinner time
            1.2,  # 8pm - evening
            1.0,  # 9pm - evening
            0.7,  # 10pm - late evening
            0.4,  # 11pm - night
        ]
        
        # Weekend pattern has more midday activity and evening social peaks
        weekend_pattern = [
            0.5,  # 12am - higher nightlife
            0.4,  # 1am - higher nightlife
            0.3,  # 2am - higher nightlife
            0.2,  # 3am - declining nightlife
            0.1,  # 4am - lowest
            0.1,  # 5am - lowest
            0.2,  # 6am - early risers
            0.3,  # 7am - morning
            0.5,  # 8am - morning
            0.8,  # 9am - mid-morning
            1.2,  # 10am - late morning
            1.5,  # 11am - approaching lunch
            1.7,  # 12pm - lunch peak
            1.6,  # 1pm - early afternoon
            1.5,  # 2pm - mid-afternoon
            1.4,  # 3pm - mid-afternoon
            1.3,  # 4pm - mid-afternoon
            1.4,  # 5pm - early evening
            1.6,  # 6pm - dinner time
            1.8,  # 7pm - evening peak
            1.9,  # 8pm - evening peak
            1.6,  # 9pm - evening
            1.3,  # 10pm - late evening
            0.8,  # 11pm - night
        ]
        
        # Friday pattern is a hybrid with commuter peaks and higher evening activity
        friday_pattern = [
            0.3,  # 12am - higher nightlife
            0.2,  # 1am - higher nightlife
            0.1,  # 2am - declining nightlife
            0.1,  # 3am - lowest
            0.2,  # 4am - very low
            0.5,  # 5am - early commuters
            1.0,  # 6am - commute building
            1.8,  # 7am - morning peak
            2.0,  # 8am - highest peak
            1.7,  # 9am - commute trailing
            1.0,  # 10am - mid-morning
            1.0,  # 11am - mid-day
            1.3,  # 12pm - lunch rush
            1.2,  # 1pm - post-lunch
            1.0,  # 2pm - mid-afternoon
            1.0,  # 3pm - mid-afternoon
            1.2,  # 4pm - early commute
            1.8,  # 5pm - evening commute starts
            1.9,  # 6pm - evening peak
            1.8,  # 7pm - dinner time
            1.7,  # 8pm - higher evening
            1.6,  # 9pm - higher evening
            1.5,  # 10pm - higher late evening
            1.0,  # 11pm - higher night
        ]
        
        # Holiday pattern resembles weekend but with less pronounced peaks
        holiday_pattern = [
            0.4,  # 12am - nightlife
            0.3,  # 1am - nightlife
            0.2,  # 2am - nightlife
            0.1,  # 3am - lowest
            0.1,  # 4am - lowest
            0.1,  # 5am - very low
            0.2,  # 6am - early risers
            0.3,  # 7am - morning
            0.5,  # 8am - morning
            0.8,  # 9am - mid-morning
            1.1,  # 10am - late morning
            1.3,  # 11am - approaching lunch
            1.5,  # 12pm - lunch
            1.4,  # 1pm - early afternoon
            1.3,  # 2pm - mid-afternoon
            1.2,  # 3pm - mid-afternoon
            1.2,  # 4pm - mid-afternoon
            1.3,  # 5pm - early evening
            1.5,  # 6pm - dinner time
            1.6,  # 7pm - evening
            1.7,  # 8pm - evening peak
            1.5,  # 9pm - evening
            1.2,  # 10pm - late evening
            0.7,  # 11pm - night
        ]
        
        return {
            "weekday": weekday_pattern,
            "friday": friday_pattern,
            "weekend": weekend_pattern,
            "holiday": holiday_pattern
        }
    
    def _create_daily_factors(self):
        """Create day of week adjustment factors"""
        return {
            0: 1.1,   # Monday: 110% of typical weekday
            1: 0.9,   # Tuesday: a bit quiet
            2: 1.05,  # Wednesday: slightly higher
            3: 1.1,   # Thursday: higher
            4: 1.3,   # Friday: significantly higher
            5: 1.2,   # Saturday: weekend high
            6: 0.9    # Sunday: wind-off before new week
        }
    
    def _create_monthly_factors(self):
        """Create monthly adjustment factors"""
        return {
            1: 0.85,  # January: post-holiday lull
            2: 0.9,   # February: winter
            3: 0.95,  # March: slight increase
            4: 1.0,   # April: baseline
            5: 1.05,  # May: improving weather
            6: 1.1,   # June: summer begins
            7: 1.2,   # July: summer peak
            8: 1.1,  # August: summer
            9: 1.2,  # September: back to school craze
            10: 1.0,  # October: baseline
            11: 0.95, # November: pre-holiday
            12: 1.15  # December: holiday season
        }
    
    def is_holiday(self, date):
        """Check if a date is a holiday"""
        for holiday in self.holidays:
            if date.year == holiday.year and date.month == holiday.month and date.day == holiday.day:
                return True
        return False
    
    def get_pattern_type(self, dt):
        """Determine the pattern type for a given datetime"""
        if self.is_holiday(dt):
            return "holiday"
        
        day_of_week = dt.weekday()
        if day_of_week == 4:  # Friday
            return "friday"
        elif day_of_week >= 5:  # Weekend
            return "weekend"
        else:  # Weekday
            return "weekday"
    
    def get_demand_multiplier(self, dt, include_noise=True):
        """
        Calculate the demand multiplier for a specific datetime.
        
        Args:
            dt: Datetime object
            include_noise: Whether to add random noise to the multiplier
        
        Returns:
            float: Demand multiplier relative to the base demand
        """
        # Get the appropriate pattern
        pattern_type = self.get_pattern_type(dt)
        hourly_pattern = self.hourly_patterns[pattern_type]
        
        # Get hourly factor (0-23)
        hour = dt.hour
        hourly_factor = hourly_pattern[hour]
        
        # Apply day of week adjustment
        day_factor = self.day_of_week_factors[dt.weekday()]
        
        # Apply monthly adjustment
        month_factor = self.monthly_factors[dt.month]
        
        # Calculate base multiplier
        multiplier = hourly_factor * day_factor * month_factor
        
        # Special events can override the normal pattern
        for event in self.special_events:
            if event["start_time"] <= dt <= event["end_time"]:
                # Apply event-specific multiplier
                if "area_specific" in event and event["area_specific"]:
                    # This event only affects specific areas
                    # Just flag it for now - geographic model will handle it
                    pass
                else:
                    # Global event impact
                    multiplier *= event["demand_multiplier"]
        
        # Add some noise to make the data more realistic
        if include_noise:
            # Use different scales of noise depending on the base demand
            if multiplier < 0.5:
                # Low demand periods have relatively more variation
                noise = np.random.normal(0, 0.1)
            elif multiplier < 1.0:
                # Medium-low demand
                noise = np.random.normal(0, 0.15)
            elif multiplier < 1.5:
                # Medium demand
                noise = np.random.normal(0, 0.2)
            else:
                # High demand periods have more absolute variation
                noise = np.random.normal(0, 0.25)
            
            multiplier = max(0.05, multiplier + noise)  # Ensure we don't go negative
        
        return multiplier
    
    def add_special_event(self, name, start_time, end_time, demand_multiplier, area_specific=False, area_name=None):
        """
        Add a special event that affects demand.
        
        Args:
            name: Event name
            start_time: Start datetime
            end_time: End datetime
            demand_multiplier: How much to multiply the normal demand by
            area_specific: Whether the event only affects a specific area
            area_name: Name of the affected area (if area_specific is True)
        """
        self.special_events.append({
            "name": name,
            "start_time": start_time,
            "end_time": end_time,
            "demand_multiplier": demand_multiplier,
            "area_specific": area_specific,
            "area_name": area_name
        })
    
    def estimate_ride_count(self, start_time, end_time, granularity_minutes=60):
        """
        Estimate the number of rides in a time period.
        
        Args:
            start_time: Start datetime
            end_time: End datetime
            granularity_minutes: Time granularity in minutes
            
        Returns:
            list: Estimated ride counts for each time interval
        """
        current_time = start_time
        ride_counts = []
        
        while current_time < end_time:
            interval_end = current_time + timedelta(minutes=granularity_minutes)
            
            # Get the average multiplier for this interval
            sample_points = 4  # Sample at 15-minute intervals
            multipliers = []
            
            for i in range(sample_points):
                sample_time = current_time + timedelta(minutes=i * granularity_minutes / sample_points)
                if sample_time < end_time:
                    multipliers.append(self.get_demand_multiplier(sample_time, include_noise=False))
            
            avg_multiplier = sum(multipliers) / len(multipliers)
            
            # Calculate expected ride count for this interval
            interval_rides = int(self.base_demand * avg_multiplier * (granularity_minutes / 60))
            
            # Add some randomness to the actual ride count
            std_dev = max(1, int(interval_rides * 0.1))  # 10% standard deviation
            actual_rides = int(np.random.normal(interval_rides, std_dev))
            actual_rides = max(0, actual_rides)  # Ensure non-negative
            
            ride_counts.append({
                "start_time": current_time,
                "end_time": interval_end,
                "expected_rides": interval_rides,
                "actual_rides": actual_rides
            })
            
            current_time = interval_end
        
        return ride_counts
    
    def save_to_json(self, filename):
        """Save the demand model to a JSON file"""
        # Convert special events to string representation first
        serializable_events = []
        for event in self.special_events:
            serializable_event = event.copy()
            serializable_event["start_time"] = event["start_time"].isoformat()
            serializable_event["end_time"] = event["end_time"].isoformat()
            serializable_events.append(serializable_event)
        
        # Serialize the model data
        model_data = {
            "base_demand": self.base_demand,
            "hourly_patterns": self.hourly_patterns,
            "day_of_week_factors": self.day_of_week_factors,
            "monthly_factors": self.monthly_factors,
            "special_events": serializable_events,
            "holidays": [h.isoformat() for h in self.holidays]
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        return filename


class TrafficModel:
    """
    Models time-based traffic patterns affecting ride durations and ETAs.
    """
    
    def __init__(self, city_name="San Francisco"):
        """
        Initialize the traffic model.
        
        Args:
            city_name: Name of the city for city-specific patterns
        """
        self.city_name = city_name
        
        # Traffic levels for different hours of the day
        self.hourly_traffic_levels = self._create_hourly_traffic_patterns()
        self.day_adjustments = self._create_day_adjustments()
        
        # Weather and special event impacts
        self.weather_conditions = {}
        self.special_events = []
    
    def _create_hourly_traffic_patterns(self):
        """Create hourly traffic level patterns"""
        # Traffic levels from 0 (free flowing) to 1 (gridlock)
        weekday_pattern = [
            0.1,  # 12am - minimal traffic
            0.05, # 1am - minimal traffic
            0.05, # 2am - minimal traffic
            0.05, # 3am - minimal traffic
            0.1,  # 4am - very light traffic
            0.2,  # 5am - building
            0.4,  # 6am - morning commute begins
            0.7,  # 7am - heavy morning rush
            0.9,  # 8am - peak morning rush
            0.7,  # 9am - rush tapering
            0.5,  # 10am - mid-morning
            0.5,  # 11am - mid-day
            0.6,  # 12pm - lunch rush
            0.5,  # 1pm - post-lunch
            0.5,  # 2pm - mid-afternoon
            0.5,  # 3pm - schools out
            0.6,  # 4pm - evening commute begins
            0.8,  # 5pm - heavy evening rush
            0.9,  # 6pm - peak evening rush 
            0.7,  # 7pm - rush tapering
            0.5,  # 8pm - evening
            0.4,  # 9pm - evening
            0.3,  # 10pm - late evening
            0.2,  # 11pm - night
        ]
        
        weekend_pattern = [
            0.2,  # 12am - nightlife traffic
            0.2,  # 1am - nightlife traffic
            0.1,  # 2am - declining nightlife
            0.05, # 3am - minimal traffic
            0.05, # 4am - minimal traffic
            0.05, # 5am - minimal traffic
            0.1,  # 6am - very light traffic
            0.2,  # 7am - light traffic
            0.3,  # 8am - building
            0.4,  # 9am - mid-morning
            0.5,  # 10am - shopping/activities begin
            0.6,  # 11am - shopping/activities
            0.7,  # 12pm - lunch/shopping peak
            0.7,  # 1pm - afternoon activities
            0.6,  # 2pm - afternoon activities
            0.6,  # 3pm - afternoon activities
            0.6,  # 4pm - afternoon activities
            0.6,  # 5pm - evening begins
            0.6,  # 6pm - dinner time
            0.5,  # 7pm - evening
            0.5,  # 8pm - evening activities
            0.4,  # 9pm - evening
            0.3,  # 10pm - late evening
            0.3,  # 11pm - nightlife begins
        ]
        
        return {
            "weekday": weekday_pattern,
            "weekend": weekend_pattern
        }
    
    def _create_day_adjustments(self):
        """Create traffic adjustment factors for different days"""
        return {
            0: 0.9,   # Monday: 90% of typical weekday
            1: 1.0,   # Tuesday: baseline weekday
            2: 1.0,   # Wednesday: typical
            3: 1.05,  # Thursday: slightly higher
            4: 1.2,   # Friday: significantly higher
            5: 0.8,   # Saturday: moderate weekend
            6: 0.7    # Sunday: lightest traffic
        }
    
    def get_traffic_level(self, dt):
        """
        Get the traffic level for a specific datetime.
        
        Args:
            dt: Datetime object
            
        Returns:
            float: Traffic level from 0 (free flowing) to 1 (gridlock)
        """
        # Determine pattern type
        day_of_week = dt.weekday()
        pattern_type = "weekend" if day_of_week >= 5 else "weekday"
        
        # Get the base traffic level
        hour = dt.hour
        base_level = self.hourly_traffic_levels[pattern_type][hour]
        
        # Apply day adjustment
        day_factor = self.day_adjustments[day_of_week]
        
        # Calculate the adjusted level
        traffic_level = base_level * day_factor
        
        # Apply weather impacts if applicable
        date_key = dt.strftime("%Y-%m-%d")
        if date_key in self.weather_conditions:
            weather = self.weather_conditions[date_key]
            if weather["type"] == "rain":
                traffic_level *= 1.2  # Rain increases traffic by 20%
            elif weather["type"] == "snow":
                traffic_level *= 1.5  # Snow increases traffic by 50%
            elif weather["type"] == "severe":
                traffic_level *= 2.0  # Severe weather can double traffic
        
        # Apply special event impacts
        for event in self.special_events:
            if event["start_time"] <= dt <= event["end_time"]:
                traffic_level *= event["traffic_multiplier"]
        
        # Add some noise (standard deviation of 10%)
        noise = np.random.normal(0, 0.1)
        traffic_level = max(0.05, min(1.0, traffic_level + noise))
        
        return traffic_level
    
    def get_traffic_descriptor(self, traffic_level):
        """Convert numeric traffic level to a descriptive category"""
        if traffic_level < 0.3:
            return "LOW"
        elif traffic_level < 0.6:
            return "MEDIUM"
        elif traffic_level < 0.8:
            return "HIGH"
        else:
            return "SEVERE"
    
    def get_trip_delay_factor(self, traffic_level):
        """Calculate how much a trip will be delayed by traffic"""
        # At traffic level 0, no delay (factor = 1)
        # At traffic level 1, trips take 2.5x longer
        return 1 + 1.5 * traffic_level
    
    def add_weather_condition(self, date, weather_type, intensity=1.0):
        """
        Add a weather condition for a specific date.
        
        Args:
            date: Date object or string in YYYY-MM-DD format
            weather_type: Type of weather ("rain", "snow", "severe")
            intensity: Intensity factor (1.0 = normal)
        """
        if isinstance(date, datetime):
            date_key = date.strftime("%Y-%m-%d")
        else:
            date_key = date
        
        self.weather_conditions[date_key] = {
            "type": weather_type,
            "intensity": intensity
        }
    
    def add_special_event(self, name, start_time, end_time, traffic_multiplier):
        """
        Add a special event that affects traffic.
        
        Args:
            name: Event name
            start_time: Start datetime
            end_time: End datetime
            traffic_multiplier: How much to multiply normal traffic by
        """
        self.special_events.append({
            "name": name,
            "start_time": start_time,
            "end_time": end_time,
            "traffic_multiplier": traffic_multiplier
        })
    
    def save_to_json(self, filename):
        """Save the traffic model to a JSON file"""
        # Convert special events to string representation first
        serializable_events = []
        for event in self.special_events:
            serializable_event = event.copy()
            serializable_event["start_time"] = event["start_time"].isoformat()
            serializable_event["end_time"] = event["end_time"].isoformat()
            serializable_events.append(serializable_event)
        
        # Serialize the model data
        model_data = {
            "city_name": self.city_name,
            "hourly_traffic_levels": self.hourly_traffic_levels,
            "day_adjustments": self.day_adjustments,
            "weather_conditions": self.weather_conditions,
            "special_events": serializable_events
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        return filename


# Function to generate Spanish holidays for a given year
def generate_spanish_holidays(year):
    """Generate common Spanish holidays for a given year"""
    holidays = []
    
    # Año Nuevo (New Year's Day)
    holidays.append(datetime(year, 1, 1))
    
    # Día de Reyes (Epiphany)
    holidays.append(datetime(year, 1, 6))
    
    # Calculate Easter Sunday (needed for several holidays)
    # Using Butcher's algorithm for Easter calculation
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    
    easter = datetime(year, month, day)
    
    # Viernes Santo (Good Friday)
    good_friday = easter - timedelta(days=2)
    holidays.append(good_friday)
    
    # Jueves Santo (Maundy Thursday) - regional but widely observed
    maundy_thursday = easter - timedelta(days=3)
    holidays.append(maundy_thursday)
    
    # Día del Trabajador (Labor Day)
    holidays.append(datetime(year, 5, 1))
    
    # Asunción de la Virgen (Assumption Day)
    holidays.append(datetime(year, 8, 15))
    
    # Fiesta Nacional de España (Spanish National Day)
    holidays.append(datetime(year, 10, 12))
    
    # Día de Todos los Santos (All Saints' Day)
    holidays.append(datetime(year, 11, 1))
    
    # Día de la Constitución (Constitution Day)
    holidays.append(datetime(year, 12, 6))
    
    # Inmaculada Concepción (Immaculate Conception)
    holidays.append(datetime(year, 12, 8))
    
    # Navidad (Christmas Day)
    holidays.append(datetime(year, 12, 25))
    
    # Nochebuena (Christmas Eve) - not official but widely observed
    holidays.append(datetime(year, 12, 24))
    
    # Nochevieja (New Year's Eve) - not official but widely observed
    holidays.append(datetime(year, 12, 31))
    
    # San José (St. Joseph's Day) - regional holiday in some communities
    holidays.append(datetime(year, 3, 19))
    
    # Santiago Apóstol (St. James Day) - important in some regions
    holidays.append(datetime(year, 7, 25))
    
    # Corpus Christi - 60 days after Easter Sunday, regional
    corpus_christi = easter + timedelta(days=60)
    holidays.append(corpus_christi)
    
    
    return holidays


# Example usage
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    parser = argparse.ArgumentParser(description='Test the temporal patterns')
    parser.add_argument('--city', type=str, default='San Francisco', help='City name')
    parser.add_argument('--base_demand', type=int, default=100, help='Base hourly demand')
    parser.add_argument('--save_demand', type=str, help='Save demand model to JSON file')
    parser.add_argument('--save_traffic', type=str, help='Save traffic model to JSON file')
    parser.add_argument('--visualize', action='store_true', help='Visualize patterns')
    
    args = parser.parse_args()
    
    # Generate holidays for the current year
    current_year = datetime.now().year
    holidays = generate_us_holidays(current_year)
    
    # Create models
    demand_model = DemandModel(base_demand=args.base_demand, holidays=holidays)
    traffic_model = TrafficModel(city_name=args.city)
    
    # Add a special event (simulating a concert)
    event_date = datetime.now() + timedelta(days=10)
    event_date = event_date.replace(hour=19, minute=0)  # 7 PM
    
    demand_model.add_special_event(
        name="Major Concert",
        start_time=event_date,
        end_time=event_date + timedelta(hours=5),
        demand_multiplier=2.5,
        area_specific=True,
        area_name="downtown"
    )
    
    traffic_model.add_special_event(
        name="Major Concert",
        start_time=event_date - timedelta(hours=2),  # Traffic starts building earlier
        end_time=event_date + timedelta(hours=6),    # And lasts longer after
        traffic_multiplier=1.8
    )
    
    # Add a weather event
    weather_date = datetime.now() + timedelta(days=5)
    traffic_model.add_weather_condition(
        date=weather_date.strftime("%Y-%m-%d"),
        weather_type="rain",
        intensity=1.2
    )
    
    if args.visualize:
        # Visualize demand pattern for a typical week
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dates = [start_date + timedelta(hours=i) for i in range(24*7)]  # One week
        
        # Calculate demand multipliers
        demand_values = [demand_model.get_demand_multiplier(dt, include_noise=False) for dt in dates]
        
        # Calculate traffic levels
        traffic_values = [traffic_model.get_traffic_level(dt) for dt in dates]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot demand
        ax1.plot(dates, demand_values)
        ax1.set_title(f'Ride Demand Pattern (Base: {args.base_demand} rides/hour)')
        ax1.set_ylabel('Demand Multiplier')
        ax1.set_ylim(0, max(demand_values) * 1.1)
        ax1.grid(True)
        
        # Format x-axis for readability
        ax1.set_xlim(dates[0], dates[-1])
        
        # Plot traffic
        ax2.plot(dates, traffic_values, color='red')
        ax2.set_title('Traffic Congestion Pattern')
        ax2.set_xlabel('Date/Time')
        ax2.set_ylabel('Traffic Level')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True)
        
        # Format x-axis
        ax2.set_xlim(dates[0], dates[-1])
        
        plt.tight_layout()
        plt.show()
    
    # Save models if requested
    if args.save_demand:
        saved_file = demand_model.save_to_json(args.save_demand)
        print(f"Demand model saved to {saved_file}")
    
    if args.save_traffic:
        saved_file = traffic_model.save_to_json(args.save_traffic)
        print(f"Traffic model saved to {saved_file}")
    
    # Test estimating ride counts
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(days=1)
    
    ride_estimates = demand_model.estimate_ride_count(start_time, end_time)
    
    print(f"\nEstimated ride counts for {start_time.date()}:")
    print("Time\t\tExpected\tActual")
    print("-" * 40)
    for estimate in ride_estimates:
        time_str = estimate["start_time"].strftime("%H:%M")
        print(f"{time_str}\t\t{estimate['expected_rides']}\t\t{estimate['actual_rides']}")
    
    total_expected = sum(e["expected_rides"] for e in ride_estimates)
    total_actual = sum(e["actual_rides"] for e in ride_estimates)
    print("-" * 40)
    print(f"Total:\t\t{total_expected}\t\t{total_actual}")
