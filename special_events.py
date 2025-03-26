"""
Special Events Generator for Ride-Hailing Simulation

This module creates special events and anomalies such as concerts, bad weather,
system outages, and fraudulent patterns to enrich the ride-hailing data with 
interesting analytics test cases.

Usage:
    from special_events import SpecialEventsGenerator
    events_generator = SpecialEventsGenerator(city_map, demand_model, traffic_model)
    events_generator.create_concert_event(event_date)
"""

import random
import uuid
import numpy as np
from datetime import datetime, timedelta
import json
import os

class SpecialEventsGenerator:
    """
    Generates special events and anomalies for ride-hailing simulation.
    """
    
    def __init__(self, city_map, demand_model, traffic_model, ride_simulator):
        """
        Initialize the special events generator.
        
        Args:
            city_map: CityMap object for geographic simulation
            demand_model: DemandModel for adjusting demand during events
            traffic_model: TrafficModel for adjusting traffic during events
            ride_simulator: RideSimulator for generating ride events
        """
        self.city_map = city_map
        self.demand_model = demand_model
        self.traffic_model = traffic_model
        self.ride_simulator = ride_simulator
        
        # Track all special events
        self.special_events = []
    
    def create_concert_event(self, event_date, venue_zone="downtown", attendees=5000, 
                            pre_event_hours=3, post_event_hours=2, name=None):
        """
        Create a concert event with high demand.
        
        Args:
            event_date: Base datetime for the event
            venue_zone: Zone in the city map where the event takes place
            attendees: Approximate number of attendees
            pre_event_hours: Hours before event with increased arrivals
            post_event_hours: Hours after event with increased departures
            name: Optional name for the event
            
        Returns:
            dict: Event details
        """
        # Set default name if none provided
        if name is None:
            name = f"Concert at {self.city_map.zones[venue_zone]['name']}"
        
        # Get venue details
        venue_location = {
            "latitude": self.city_map.zones[venue_zone]["center"][0],
            "longitude": self.city_map.zones[venue_zone]["center"][1],
            "address": f"{random.randint(100, 999)} {venue_zone.title()} Calle",
            "city": self.city_map.city_name
        }
        
        # Event start and end times
        event_start = event_date
        event_end = event_date + timedelta(hours=3)  # Typical 3-hour event
        
        # Calculate demand pattern
        # Pre-event arrivals gradually increase
        arrivals_start = event_start - timedelta(hours=pre_event_hours)
        arrivals_end = event_start + timedelta(minutes=15)  # Most arrive by 15 mins after start
        
        # Departures concentrated after event
        departures_start = event_end - timedelta(minutes=30)  # Some leave a bit early
        departures_end = event_end + timedelta(hours=post_event_hours)
        
        # Calculate ride volume
        # Assume 60-80% of attendees use the service, with 2-3 people per ride
        ride_percent = random.uniform(0.6, 0.8)
        avg_people_per_ride = random.uniform(2, 3)
        
        total_rides = int(attendees * ride_percent / avg_people_per_ride)
        arrival_rides = int(total_rides * 0.8)  # 70% use service to arrive
        departure_rides = int(total_rides * 0.9)  # 80% use service to depart (more at once)
        
        # Register event with demand and traffic models
        # Demand impact around venue
        self.demand_model.add_special_event(
            name=name,
            start_time=arrivals_start,
            end_time=departures_end,
            demand_multiplier=2.0,  # Overall demand boost
            area_specific=True,
            area_name=venue_zone
        )
        
        # Traffic impact in and around venue
        self.traffic_model.add_special_event(
            name=name,
            start_time=arrivals_start,
            end_time=departures_end,
            traffic_multiplier=1.8  # Significant traffic increase
        )
        
        # Store event details
        event = {
            "type": "concert",
            "name": name,
            "venue_zone": venue_zone,
            "venue_location": venue_location,
            "event_start": str(event_start),
            "event_end": str(event_end),
            "arrivals_start": str(arrivals_start),
            "arrivals_end": str(arrivals_end),
            "departures_start": str(departures_start),
            "departures_end": str(departures_end),
            "arrival_rides": arrival_rides,
            "departure_rides": departure_rides,
            "estimated_attendees": attendees
        }
        
        self.special_events.append(event)
        return event
    
    def create_sports_event(self, event_date, venue_zone="downtown", attendees=20000, 
                          pre_event_hours=2, post_event_hours=1.5, name=None):
        """
        Create a sporting event with high demand and more concentrated arrival/departure.
        
        Args:
            event_date: Base datetime for the event
            venue_zone: Zone in the city map where the event takes place
            attendees: Approximate number of attendees
            pre_event_hours: Hours before event with increased arrivals
            post_event_hours: Hours after event with increased departures
            name: Optional name for the event
            
        Returns:
            dict: Event details
        """
        # Similar implementation to concert but with more concentrated patterns
        # Set default name if none provided
        if name is None:
            teams = ["Real Madrid", "Atlético de Madrid", "Barça", "Getafe FC", "Sevilla FC", "Rayo Vallecano", "Celta Vigo"]
            name = f"{random.choice(teams)} vs {random.choice(teams)} Game"
        
        # Get venue details
        venue_location = {
            "latitude": self.city_map.zones[venue_zone]["center"][0],
            "longitude": self.city_map.zones[venue_zone]["center"][1],
            "address": f"{random.randint(100, 999)} Sports Stadium",
            "city": self.city_map.city_name
        }
        
        # Event start and end times
        event_start = event_date
        event_end = event_date + timedelta(hours=3)  # Typical 3-hour event
        
        # Calculate demand pattern - more concentrated than concerts
        # Pre-event arrivals gradually increase
        arrivals_start = event_start - timedelta(hours=pre_event_hours)
        arrivals_end = event_start + timedelta(minutes=15) 
        
        # Departures very concentrated after event
        departures_start = event_end - timedelta(minutes=15)  # Some leave a bit early
        departures_end = event_end + timedelta(hours=post_event_hours)
        
        # Calculate ride volume
        # Assume 40-60% of attendees use the service, with 2-4 people per ride
        ride_percent = random.uniform(0.4, 0.6)
        avg_people_per_ride = random.uniform(2, 4)
        
        total_rides = int(attendees * ride_percent / avg_people_per_ride)
        arrival_rides = int(total_rides * 0.7)  # 70% use service to arrive
        departure_rides = int(total_rides * 0.9)  # 90% use service to depart (more at once)
        
        # Register event with demand and traffic models
        # Demand impact around venue
        self.demand_model.add_special_event(
            name=name,
            start_time=arrivals_start,
            end_time=departures_end,
            demand_multiplier=2.5,  # Higher than concerts - more concentrated
            area_specific=True,
            area_name=venue_zone
        )
        
        # Traffic impact in and around venue
        self.traffic_model.add_special_event(
            name=name,
            start_time=arrivals_start,
            end_time=departures_end,
            traffic_multiplier=2.2  # More severe than concerts
        )
        
        # Store event details
        event = {
            "type": "sports",
            "name": name,
            "venue_zone": venue_zone,
            "venue_location": venue_location,
            "event_start": str(event_start),
            "event_end": str(event_end),
            "arrivals_start": str(arrivals_start),
            "arrivals_end": str(arrivals_end),
            "departures_start": str(departures_start),
            "departures_end": str(departures_end),
            "arrival_rides": arrival_rides,
            "departure_rides": departure_rides,
            "estimated_attendees": attendees
        }
        
        self.special_events.append(event)
        return event
    
    def create_weather_event(self, event_date, duration_hours=8, severity="medium", affected_zones=None):
        """
        Create a weather event that affects traffic and ride patterns.
        
        Args:
            event_date: Start datetime for the weather event
            duration_hours: How long the weather event lasts
            severity: "light", "medium", "severe"
            affected_zones: List of zones affected (None for all)
            
        Returns:
            dict: Event details
        """
        # Determine weather type and impact factors
        weather_types = {
            "light": {
                "types": ["light_rain", "fog"],
                "traffic_factor": 1.2,
                "demand_factor": 1.3,
                "cancellation_increase": 0.05
            },
            "medium": {
                "types": ["rain", "snow"],
                "traffic_factor": 1.5,
                "demand_factor": 1.5,
                "cancellation_increase": 0.1
            },
            "severe": {
                "types": ["heavy_rain", "snowstorm", "hail"],
                "traffic_factor": 2.0,
                "demand_factor": 2.0,
                "cancellation_increase": 0.2
            }
        }
        
        weather_impact = weather_types[severity]
        weather_type = random.choice(weather_impact["types"])
        
        # Weather event time range
        event_start = event_date
        event_end = event_date + timedelta(hours=duration_hours)
        
        # Set affected zones
        if affected_zones is None:
            affected_zones = list(self.city_map.zones.keys())
        
        # Register with traffic model
        self.traffic_model.add_weather_condition(
            date=event_date.strftime("%Y-%m-%d"),
            weather_type=weather_type,
            intensity=weather_impact["traffic_factor"]
        )
        
        # Register with demand model - weather increases ride demand
        self.demand_model.add_special_event(
            name=f"{weather_type.replace('_', ' ').title()} Weather",
            start_time=event_start,
            end_time=event_end,
            demand_multiplier=weather_impact["demand_factor"]
        )
        
        # Store event details
        event = {
            "type": "weather",
            "weather_type": weather_type,
            "severity": severity,
            "event_start": event_start,
            "event_end": event_end,
            "affected_zones": affected_zones,
            "traffic_factor": weather_impact["traffic_factor"],
            "demand_factor": weather_impact["demand_factor"],
            "cancellation_increase": weather_impact["cancellation_increase"]
        }
        
        self.special_events.append(event)
        return event
    
    def create_system_outage(self, event_date, duration_minutes=90, severity="partial"):
        """
        Create a system outage or disruption event.
        
        Args:
            event_date: Start datetime for the outage
            duration_minutes: How long the outage lasts
            severity: "partial" or "complete"
            
        Returns:
            dict: Event details
        """
        # Outage time range
        event_start = event_date
        event_end = event_date + timedelta(minutes=duration_minutes)
        
        # Determine impact based on severity
        if severity == "complete":
            # Complete outage - no rides possible
            failure_rate = 1.0
            cancellation_rate = 1.0
        else:
            # Partial outage - some rides still go through
            failure_rate = random.uniform(0.6, 0.9)
            cancellation_rate = random.uniform(0.4, 0.7)
        
        # Store event details
        event = {
            "type": "system_outage",
            "severity": severity,
            "event_start": event_start,
            "event_end": event_end,
            "duration_minutes": duration_minutes,
            "failure_rate": failure_rate,
            "cancellation_rate": cancellation_rate
        }
        
        self.special_events.append(event)
        return event
    
    def create_fraud_patterns(self, start_date, end_date, num_fraud_users=5, num_fraud_drivers=3):
        """
        Create fraudulent ride patterns for analytics detection.
        
        Args:
            start_date: Start of the date range to inject fraud
            end_date: End of the date range
            num_fraud_users: Number of users to create fraud patterns for
            num_fraud_drivers: Number of drivers to create fraud patterns for
            
        Returns:
            list: Fraudulent pattern events
        """
        fraud_events = []
        
        # Define fraud pattern types
        fraud_patterns = [
            {
                "name": "circular_rides",
                "description": "Rides that start and end at nearly the same location",
                "avg_rides_per_user": random.randint(3, 8)
            },
            {
                "name": "impossible_trips",
                "description": "Sequential rides with impossible timing",
                "avg_rides_per_user": random.randint(2, 5)
            },
            {
                "name": "fake_surge",
                "description": "Artificial surge creation through fake requests",
                "avg_rides_per_user": random.randint(15, 30)
            },
            {
                "name": "rating_manipulation",
                "description": "Artificially manipulating ratings",
                "avg_rides_per_user": random.randint(8, 15)
            }
        ]
        
        # Select users and drivers for fraud
        if hasattr(self.ride_simulator, 'users') and len(self.ride_simulator.users) >= num_fraud_users:
            fraud_users = random.sample(self.ride_simulator.users, num_fraud_users)
        else:
            # Create fake users if simulator doesn't have enough
            fraud_users = []
            for i in range(num_fraud_users):
                fraud_users.append({
                    "user_id": f"F-U{str(i).zfill(5)}",
                    "cancellation_rate": random.uniform(0.1, 0.3),
                    "platform": random.choice(["iOS", "Android"])
                })
        
        if hasattr(self.ride_simulator, 'drivers') and len(self.ride_simulator.drivers) >= num_fraud_drivers and num_fraud_drivers > 1:
            fraud_drivers = random.sample(self.ride_simulator.drivers, num_fraud_drivers)
        else:
            # Create fake drivers if simulator doesn't have enough
            fraud_drivers = []
            if num_fraud_drivers == 1:
                for i in range(2):
                    fraud_drivers.append({
                        "driver_id": f"F-D{str(i).zfill(5)}",
                        "rating": random.uniform(4.0, 4.9),
                        "cancellation_rate": random.uniform(0.05, 0.15)
                    })
            else:
                for i in range(num_fraud_drivers):
                    fraud_drivers.append({
                        "driver_id": f"F-D{str(i).zfill(5)}",
                        "rating": random.uniform(4.0, 4.9),
                        "cancellation_rate": random.uniform(0.05, 0.15)
                    })
        
        # Store details for later query analysis
        fraud_details = {
            "type": "fraud_patterns",
            "start_date": start_date,
            "end_date": end_date,
            "fraud_users": [u["user_id"] for u in fraud_users],
            "fraud_drivers": [d["driver_id"] for d in fraud_drivers],
            "patterns": [p["name"] for p in fraud_patterns],
            "events": []
        }
        
        # For each fraud user, create some patterns
        for user in fraud_users:
            # Pick 1-3 fraud patterns for this user
            user_patterns = random.sample(fraud_patterns, random.randint(1, 3))
            
            for pattern in user_patterns:
                # Number of rides in this pattern
                num_rides = max(1, int(random.normalvariate(
                    pattern["avg_rides_per_user"], 
                    pattern["avg_rides_per_user"] / 4
                )))
                
                # Spread rides throughout the date range
                total_days = (end_date - start_date).days
                
                for _ in range(num_rides):
                    # Pick a random day and time
                    fraud_day = start_date + timedelta(days=random.randint(0, total_days-1))
                    fraud_time = datetime.combine(
                        fraud_day.date(),
                        datetime.min.time()
                    ) + timedelta(hours=random.randint(8, 22))
                    
                    # Generate the specific fraud pattern
                    if pattern["name"] == "circular_rides":
                        fraud_event = self._generate_circular_ride(user, fraud_time, fraud_drivers)
                    elif pattern["name"] == "collusion":
                        # Always use the same driver for this user
                        colluding_driver = random.choice(fraud_drivers)
                        fraud_event = self._generate_collusion_ride(user, colluding_driver, fraud_time)
                    elif pattern["name"] == "impossible_trips":
                        fraud_event = self._generate_impossible_trip(user, fraud_time, fraud_drivers)
                    elif pattern["name"] == "fake_surge":
                        fraud_event = self._generate_fake_surge(user, fraud_time)
                    elif pattern["name"] == "rating_manipulation":
                        fraud_event = self._generate_rating_manipulation(user, fraud_time, fraud_drivers)
                    
                    if fraud_event:
                        fraud_events.extend(fraud_event)
                        fraud_details["events"].append({
                            "pattern": pattern["name"],
                            "user_id": user["user_id"],
                            "timestamp": fraud_time.isoformat(),
                            "event_ids": [e["event_id"] for e in fraud_event if "event_id" in e]
                        })
        
        self.special_events.append(fraud_details)
        return fraud_events
    
    def _generate_circular_ride(self, user, time, fraud_drivers):
        """Generate a circular ride that starts and ends at nearly the same location"""
        driver = random.choice(fraud_drivers)
        
        # Get a location
        origin_location = self.city_map.generate_location(time, "origin")
        
        # Create a destination very close to the origin
        destination_location = {
            "latitude": origin_location["latitude"] + random.uniform(-0.001, 0.001),
            "longitude": origin_location["longitude"] + random.uniform(-0.001, 0.001),
            "address": f"{random.randint(100, 999)} {origin_location['address'].split(' ', 1)[1]}",
            "city": origin_location["city"]
        }
        
        # Calculate a very short distance
        distance_km = max(0.5, self.city_map.calculate_distance(origin_location, destination_location))
        
        # Generate inflated fare
        base_fare = 5.0 + distance_km * 1.2
        surge_multiplier = random.uniform(1.8, 3.0)  # Artificially high surge
        total_fare = base_fare * surge_multiplier
        
        # Create event sequence
        ride_id = f"F-R{uuid.uuid4().hex[:8]}"
        session_id = f"F-S{uuid.uuid4().hex[:8]}"
        event_id_base = f"F-E{uuid.uuid4().hex[:8]}"
        current_timestamp = int(time.timestamp() * 1000)
        
        events = []
        
        # RIDE_REQUESTED
        events.append({
            "event_id": f"{event_id_base}_1",
            "ride_id": ride_id,
            "event_type": "RIDE_REQUESTED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": None,
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": {
                "distance_km": distance_km,
                "estimated_duration_minutes": int(distance_km * 3 + 5),
                "actual_duration_minutes": None,
                "vehicle_type": "Economy",
                "base_fare": round(base_fare, 2),
                "surge_multiplier": surge_multiplier,
                "total_fare": round(total_fare, 2)
            },
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # DRIVER_ASSIGNED - quick assignment
        current_timestamp += random.randint(5000, 15000)
        events.append({
            "event_id": f"{event_id_base}_2",
            "ride_id": ride_id,
            "event_type": "DRIVER_ASSIGNED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": events[0]["ride_details"],
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # Skip to RIDE_COMPLETED - very quick ride
        current_timestamp += random.randint(300000, 900000)  # 5-15 minutes
        
        ride_details = events[0]["ride_details"].copy()
        ride_details["actual_duration_minutes"] = int((current_timestamp - events[0]["timestamp"]) / 60000)
        
        events.append({
            "event_id": f"{event_id_base}_3",
            "ride_id": ride_id,
            "event_type": "RIDE_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": ride_details,
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # PAYMENT_COMPLETED
        current_timestamp += random.randint(5000, 15000)  # 5-15 seconds
        events.append({
            "event_id": f"{event_id_base}_4",
            "ride_id": ride_id,
            "event_type": "PAYMENT_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": ride_details,
            "payment_info": {
                "payment_method": "CARD",
                "payment_status": "COMPLETED",
                "payment_id": f"P-{uuid.uuid4().hex[:10]}"
            },
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        return events
    
    def _generate_collusion_ride(self, user, driver, time):
        """Generate a ride where the same user and driver are repeatedly matched"""
        # Similar to circular ride but with focus on the collusion aspect
        # Get locations
        origin_location = self.city_map.generate_location(time, "origin")
        destination_location = self.city_map.generate_location(time, "destination", origin=origin_location)
        
        # Calculate distance
        distance_km = self.city_map.calculate_distance(origin_location, destination_location)
        
        # Generate inflated fare
        base_fare = 5.0 + distance_km * 1.2
        surge_multiplier = random.uniform(1.5, 2.5)  # Higher than normal
        total_fare = base_fare * surge_multiplier
        
        # Create event sequence similar to circular ride
        ride_id = f"F-R{uuid.uuid4().hex[:8]}"
        session_id = f"F-S{uuid.uuid4().hex[:8]}"
        event_id_base = f"F-E{uuid.uuid4().hex[:8]}"
        current_timestamp = int(time.timestamp() * 1000)
        
        events = []
        
        # RIDE_REQUESTED
        events.append({
            "event_id": f"{event_id_base}_1",
            "ride_id": ride_id,
            "event_type": "RIDE_REQUESTED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": None,
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": {
                "distance_km": distance_km,
                "estimated_duration_minutes": int(distance_km * 2 + 5),
                "actual_duration_minutes": None,
                "vehicle_type": "Economy",
                "base_fare": round(base_fare, 2),
                "surge_multiplier": surge_multiplier,
                "total_fare": round(total_fare, 2)
            },
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # DRIVER_ASSIGNED - very quick assignment to colluding driver
        current_timestamp += random.randint(2000, 8000)  # 2-8 seconds (suspiciously fast)
        events.append({
            "event_id": f"{event_id_base}_2",
            "ride_id": ride_id,
            "event_type": "DRIVER_ASSIGNED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],  # Always the same driver
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": events[0]["ride_details"],
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # DRIVER_ARRIVED - very quick arrival
        current_timestamp += random.randint(60000, 180000)  # 1-3 minutes
        events.append({
            "event_id": f"{event_id_base}_3",
            "ride_id": ride_id,
            "event_type": "DRIVER_ARRIVED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": events[0]["ride_details"],
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # RIDE_STARTED
        current_timestamp += random.randint(30000, 90000)  # 30-90 seconds
        events.append({
            "event_id": f"{event_id_base}_4",
            "ride_id": ride_id,
            "event_type": "RIDE_STARTED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": events[0]["ride_details"],
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # RIDE_COMPLETED - with typical duration
        actual_duration_minutes = int(distance_km * 2)  # ~30 km/h
        current_timestamp += actual_duration_minutes * 60 * 1000
        
        ride_details = events[0]["ride_details"].copy()
        ride_details["actual_duration_minutes"] = actual_duration_minutes
        
        events.append({
            "event_id": f"{event_id_base}_5",
            "ride_id": ride_id,
            "event_type": "RIDE_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": ride_details,
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # PAYMENT_COMPLETED
        current_timestamp += random.randint(5000, 15000)  # 5-15 seconds
        events.append({
            "event_id": f"{event_id_base}_6",
            "ride_id": ride_id,
            "event_type": "PAYMENT_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": ride_details,
            "payment_info": {
                "payment_method": "CARD",
                "payment_status": "COMPLETED",
                "payment_id": f"P-{uuid.uuid4().hex[:10]}"
            },
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # Both give perfect ratings
        current_timestamp += random.randint(5000, 30000)  # 5-30 seconds
        events.append({
            "event_id": f"{event_id_base}_7",
            "ride_id": ride_id,
            "event_type": "USER_RATED_DRIVER",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "ratings": {
                "user_to_driver_rating": 5.0,
                "driver_to_user_rating": None,
                "user_comment": "Great driver!",
                "driver_comment": None
            },
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        current_timestamp += random.randint(5000, 30000)  # 5-30 seconds
        events.append({
            "event_id": f"{event_id_base}_8",
            "ride_id": ride_id,
            "event_type": "DRIVER_RATED_USER",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "ratings": {
                "user_to_driver_rating": 5.0,
                "driver_to_user_rating": 5.0,
                "user_comment": "Great driver!",
                "driver_comment": "Perfect passenger!"
            },
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        return events
    
    def _generate_impossible_trip(self, user, time, fraud_drivers):
        """Generate sequential rides with impossible timing (too close together)"""
        # Create two rides back to back with impossible timing
        driver1 = random.choice(fraud_drivers)
        driver2 = random.choice([d for d in fraud_drivers if d != driver1])
        
        # First ride
        origin1 = self.city_map.generate_location(time, "origin")
        destination1 = self.city_map.generate_location(time, "destination", origin=origin1)
        
        # Second ride starts at the first destination but has impossible timing
        origin2 = destination1
        destination2 = self.city_map.generate_location(time, "destination", origin=origin2)
        
        # Calculate distances
        distance1 = self.city_map.calculate_distance(origin1, destination1)
        distance2 = self.city_map.calculate_distance(origin2, destination2)
        
        # Generate fares
        base_fare1 = 5.0 + distance1 * 1.2
        base_fare2 = 5.0 + distance2 * 1.2
        
        surge_multiplier = random.uniform(1.2, 2.0)
        total_fare1 = base_fare1 * surge_multiplier
        total_fare2 = base_fare2 * surge_multiplier
        
        # Create first ride
        ride_id1 = f"F-R{uuid.uuid4().hex[:8]}"
        session_id = f"F-S{uuid.uuid4().hex[:8]}"
        event_id_base1 = f"F-E{uuid.uuid4().hex[:8]}"
        current_timestamp = int(time.timestamp() * 1000)
        
        events = []
        
        # First ride events (simplified)
        events.append({
            "event_id": f"{event_id_base1}_1",
            "ride_id": ride_id1,
            "event_type": "RIDE_REQUESTED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": None,
            "pickup_location": origin1,
            "dropoff_location": destination1,
            "ride_details": {
                "distance_km": distance1,
                "estimated_duration_minutes": int(distance1 * 2 + 5),
                "actual_duration_minutes": None,
                "vehicle_type": "Economy",
                "base_fare": round(base_fare1, 2),
                "surge_multiplier": surge_multiplier,
                "total_fare": round(total_fare1, 2)
            },
            "app_version": "4.6.0",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # Skip details and just add completed ride
        ride_duration1 = int(distance1 * 2)  # minutes
        current_timestamp += (ride_duration1 + random.randint(5, 15)) * 60 * 1000  # Add some buffer time
        
        ride_details1 = events[0]["ride_details"].copy()
        ride_details1["actual_duration_minutes"] = ride_duration1
        
        events.append({
            "event_id": f"{event_id_base1}_2",
            "ride_id": ride_id1,
            "event_type": "RIDE_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver1["driver_id"],
            "pickup_location": origin1,
            "dropoff_location": destination1,
            "ride_details": ride_details1,
            "app_version": "4.6.0",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # Second ride requested impossibly quickly after first ride
        # Add only 1-3 minutes - this should be impossible as the user would
        # need more time to exit the car, etc.
        current_timestamp += random.randint(60000, 180000)  # 1-3 minutes
        
        ride_id2 = f"F-R{uuid.uuid4().hex[:8]}"
        event_id_base2 = f"F-E{uuid.uuid4().hex[:8]}"
        
        events.append({
            "event_id": f"{event_id_base2}_1",
            "ride_id": ride_id2,
            "event_type": "RIDE_REQUESTED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": None,
            "pickup_location": origin2,
            "dropoff_location": destination2,
            "ride_details": {
                "distance_km": distance2,
                "estimated_duration_minutes": int(distance2 * 2 + 5),
                "actual_duration_minutes": None,
                "vehicle_type": "Economy",
                "base_fare": round(base_fare2, 2),
                "surge_multiplier": surge_multiplier,
                "total_fare": round(total_fare2, 2)
            },
            "app_version": "4.6.0",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # Second ride also completes
        ride_duration2 = int(distance2 * 2)  # minutes
        current_timestamp += (ride_duration2 + random.randint(5, 15)) * 60 * 1000
        
        ride_details2 = events[2]["ride_details"].copy()
        ride_details2["actual_duration_minutes"] = ride_duration2
        
        events.append({
            "event_id": f"{event_id_base2}_2",
            "ride_id": ride_id2,
            "event_type": "RIDE_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver2["driver_id"],
            "pickup_location": origin2,
            "dropoff_location": destination2,
            "ride_details": ride_details2,
            "app_version": "4.6.0",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        return events
    
    def _generate_fake_surge(self, user, time):
        """Generate multiple ride requests to artificially create surge pricing"""
        # Create multiple ride requests in same area in quick succession
        # These are usually not completed
        origin_location = self.city_map.generate_location(time, "origin")
        
        events = []
        current_timestamp = int(time.timestamp() * 1000)
        
        # Generate 3-5 quick requests
        num_requests = random.randint(3, 5)
        
        for i in range(num_requests):
            # Slightly vary the pickup location
            pickup = {
                "latitude": origin_location["latitude"] + random.uniform(-0.002, 0.002),
                "longitude": origin_location["longitude"] + random.uniform(-0.002, 0.002),
                "address": origin_location["address"],
                "city": origin_location["city"]
            }
            
            # Generate a random destination
            destination = self.city_map.generate_location(time, "destination")
            
            # Calculate distance
            distance_km = self.city_map.calculate_distance(pickup, destination)
            
            # Generate fare
            base_fare = 5.0 + distance_km * 1.2
            
            # Each request increases the surge multiplier
            surge_multiplier = 1.0 + (i * 0.2)
            total_fare = base_fare * surge_multiplier
            
            ride_id = f"F-R{uuid.uuid4().hex[:8]}"
            session_id = f"F-S{uuid.uuid4().hex[:8]}"
            event_id = f"F-E{uuid.uuid4().hex[:8]}"
            
            # Create request event
            events.append({
                "event_id": event_id,
                "ride_id": ride_id,
                "event_type": "RIDE_REQUESTED",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": None,
                "pickup_location": pickup,
                "dropoff_location": destination,
                "ride_details": {
                    "distance_km": distance_km,
                    "estimated_duration_minutes": int(distance_km * 2 + 5),
                    "actual_duration_minutes": None,
                    "vehicle_type": "Economy",
                    "base_fare": round(base_fare, 2),
                    "surge_multiplier": surge_multiplier,
                    "total_fare": round(total_fare, 2)
                },
                "app_version": "4.6.0",
                "platform": user.get("platform", "iOS"),
                "session_id": session_id
            })
            
            # Add cancellation for most of these requests
            if i < num_requests - 1:  # Cancel all but the last one
                cancel_timestamp = current_timestamp + random.randint(30000, 90000)  # 30-90 seconds
                
                events.append({
                    "event_id": f"{event_id}_cancel",
                    "ride_id": ride_id,
                    "event_type": "RIDE_CANCELED_BY_USER",
                    "timestamp": cancel_timestamp,
                    "user_id": user["user_id"],
                    "driver_id": None,
                    "pickup_location": pickup,
                    "dropoff_location": destination,
                    "ride_details": events[-1]["ride_details"],
                    "cancellation_info": {
                        "canceled_by": "USER",
                        "cancellation_reason": random.choice([
                            "Changed plans", 
                            "Waiting too long",
                            "Requested by mistake"
                        ]),
                        "cancellation_fee": 0.0
                    },
                    "app_version": "4.6.0",
                    "platform": user.get("platform", "iOS"),
                    "session_id": session_id
                })
            
            # Update timestamp for next request
            current_timestamp += random.randint(60000, 180000)  # 1-3 minutes
        
        return events
    
    def _generate_rating_manipulation(self, user, time, fraud_drivers):
        """Generate a ride with artificially manipulated ratings"""
        # Similar to collusion but focusing on extreme ratings
        driver = random.choice(fraud_drivers)
        
        # Get locations
        origin_location = self.city_map.generate_location(time, "origin")
        destination_location = self.city_map.generate_location(time, "destination", origin=origin_location)
        
        # Generate a typical ride first
        ride_id = f"F-R{uuid.uuid4().hex[:8]}"
        session_id = f"F-S{uuid.uuid4().hex[:8]}"
        event_id_base = f"F-E{uuid.uuid4().hex[:8]}"
        current_timestamp = int(time.timestamp() * 1000)
        
        events = []
        
        # Simplified ride events - just focus on the rating part
        # Skip to RIDE_COMPLETED
        distance_km = self.city_map.calculate_distance(origin_location, destination_location)
        ride_duration = int(distance_km * 2)  # minutes
        
        base_fare = 5.0 + distance_km * 1.2
        surge_multiplier = 1.0
        total_fare = base_fare * surge_multiplier
        
        events.append({
            "event_id": f"{event_id_base}_1",
            "ride_id": ride_id,
            "event_type": "RIDE_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": {
                "distance_km": distance_km,
                "estimated_duration_minutes": int(distance_km * 2 + 5),
                "actual_duration_minutes": ride_duration,
                "vehicle_type": "Economy",
                "base_fare": round(base_fare, 2),
                "surge_multiplier": surge_multiplier,
                "total_fare": round(total_fare, 2)
            },
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # PAYMENT_COMPLETED
        current_timestamp += random.randint(5000, 15000)  # 5-15 seconds
        events.append({
            "event_id": f"{event_id_base}_2",
            "ride_id": ride_id,
            "event_type": "PAYMENT_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": origin_location,
            "dropoff_location": destination_location,
            "ride_details": events[0]["ride_details"],
            "payment_info": {
                "payment_method": "CARD",
                "payment_status": "COMPLETED",
                "payment_id": f"P-{uuid.uuid4().hex[:10]}"
            },
            "app_version": "4.5.2",
            "platform": user.get("platform", "iOS"),
            "session_id": session_id
        })
        
        # Manipulated ratings - 50% chance of extremely high or extremely low
        rating_type = random.choice(["high", "low"])
        
        current_timestamp += random.randint(5000, 30000)  # 5-30 seconds
        
        if rating_type == "high":
            # Perfect 5-star rating with glowing comments
            events.append({
                "event_id": f"{event_id_base}_3",
                "ride_id": ride_id,
                "event_type": "USER_RATED_DRIVER",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": driver["driver_id"],
                "ratings": {
                    "user_to_driver_rating": 5.0,
                    "driver_to_user_rating": None,
                    "user_comment": random.choice([
                        "Best driver ever! Amazing service!",
                        "Absolutely perfect ride and driver!",
                        "Could not have been better! 5 stars!",
                        "Incredible driver, extremely professional!"
                    ]),
                    "driver_comment": None
                },
                "app_version": "4.5.2",
                "platform": user.get("platform", "iOS"),
                "session_id": session_id
            })
            
            current_timestamp += random.randint(5000, 30000)  # 5-30 seconds
            events.append({
                "event_id": f"{event_id_base}_4",
                "ride_id": ride_id,
                "event_type": "DRIVER_RATED_USER",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": driver["driver_id"],
                "ratings": {
                    "user_to_driver_rating": 5.0,
                    "driver_to_user_rating": 5.0,
                    "user_comment": events[-1]["ratings"]["user_comment"],
                    "driver_comment": random.choice([
                        "Perfect passenger! A pleasure to drive!",
                        "Wonderful rider, very respectful!",
                        "Excellent customer, 5 stars!",
                        "Best passenger I've had!"
                    ])
                },
                "app_version": "4.5.2",
                "platform": user.get("platform", "iOS"),
                "session_id": session_id
            })
        else:
            # Extremely negative rating (1-star)
            events.append({
                "event_id": f"{event_id_base}_3",
                "ride_id": ride_id,
                "event_type": "USER_RATED_DRIVER",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": driver["driver_id"],
                "ratings": {
                    "user_to_driver_rating": 1.0,
                    "driver_to_user_rating": None,
                    "user_comment": random.choice([
                        "Terrible driver, avoid at all costs!",
                        "Worst experience ever, very unprofessional!",
                        "Dangerous driving and rude attitude!",
                        "Car was dirty and driver was late!"
                    ]),
                    "driver_comment": None
                },
                "app_version": "4.5.2",
                "platform": user.get("platform", "iOS"),
                "session_id": session_id
            })
            
            current_timestamp += random.randint(5000, 30000)  # 5-30 seconds
            events.append({
                "event_id": f"{event_id_base}_4",
                "ride_id": ride_id,
                "event_type": "DRIVER_RATED_USER",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": driver["driver_id"],
                "ratings": {
                    "user_to_driver_rating": 1.0,
                    "driver_to_user_rating": 1.0,
                    "user_comment": events[-1]["ratings"]["user_comment"],
                    "driver_comment": random.choice([
                        "Very rude passenger!",
                        "Disrespectful and made a mess in my car!",
                        "Made me wait and was aggressive!",
                        "Worst passenger I've had!"
                    ])
                },
                "app_version": "4.5.2",
                "platform": user.get("platform", "iOS"),
                "session_id": session_id
            })
        
        return events
    
    def generate_special_events_for_period(self, start_date, end_date, output_file=None):
        """
        Generate all special events for a time period.
        
        Args:
            start_date: Start date for the simulation
            end_date: End date for the simulation
            output_file: File to save events to (optional)
            
        Returns:
            list: All generated special events
        """
        # Calculate total days in the simulation period
        days = (end_date - start_date).days

        # Determine number of events based on realistic frequencies for a 3-month period in Madrid
        num_concerts = max(1, days // 4)      # roughly one concert every 4 days
        num_sports = max(1, days // 15)         # roughly one sports event every 15 days
        num_weather = max(1, days // 10)        # roughly one weather event every 10 days
        num_fraud_patterns = max(2, days // 7)  # roughly two fraud patterns every 7 days

        # 1. Add multiple concert events
        for i in range(num_concerts):
            event_day = start_date + timedelta(days=random.randint(0, days - 1))
            event_time = datetime.combine(event_day.date(), datetime.strptime("19:00", "%H:%M").time())
            concert = self.create_concert_event(
                event_time,
                venue_zone=random.choice(list(self.city_map.zones.keys())),
                attendees=random.randint(3000, 8000),
                name=f"Concert Event {i+1}"
            )
            print(f"Added concert event: {concert['name']} on {event_time}")

        # 2. Add multiple sports events
        for i in range(num_sports):
            event_day = start_date + timedelta(days=random.randint(0, days - 1))
            event_time = datetime.combine(event_day.date(), datetime.strptime("15:00", "%H:%M").time())
            sports = self.create_sports_event(
                event_time,
                venue_zone=random.choice(list(self.city_map.zones.keys())),
                attendees=random.randint(15000, 30000),
                name=f"Sports Event {i+1}"
            )
            print(f"Added sports event: {sports['name']} on {event_time}")

        # 3. Add multiple weather events
        for i in range(num_weather):
            event_day = start_date + timedelta(days=random.randint(0, days - 1))
            event_time = datetime.combine(event_day.date(), datetime.strptime("08:00", "%H:%M").time())
            weather = self.create_weather_event(
                event_time,
                duration_hours=random.randint(6, 12),
                severity=random.choice(["medium", "severe"])
            )
            print(f"Added weather event: {weather['weather_type']} on {event_time}")

        # 4. Add a single system outage of partial type
        outage_day = start_date + timedelta(days=random.randint(0, days - 1))
        outage_time = datetime.combine(outage_day.date(), datetime.strptime(f"{random.randint(10, 18)}:00", "%H:%M").time())
        outage = self.create_system_outage(
            outage_time,
            duration_minutes=random.randint(30, 180),
            severity="partial"
        )
        print(f"Added system outage on {outage_time} for {outage['duration_minutes']} minutes")

        # 5. Add fraud patterns
        # Here we call create_fraud_patterns with the number of fraud users and drivers based on our calculated frequency.
        fraud_events = self.create_fraud_patterns(
            start_date,
            end_date,
            num_fraud_users=num_fraud_patterns,
            num_fraud_drivers=num_fraud_patterns
        )
        print(f"Added {len(fraud_events)} fraudulent ride events")

        # Return all special event metadata (for demonstration)
        return self.special_events
    
    def save_to_json(self, filename):
        """Save special events metadata to JSON file"""
        # Convert datetime objects to strings for JSON serialization
        serializable_events = []
        
        for event in self.special_events:
            serializable_event = {}
            for key, value in event.items():
                if isinstance(value, datetime):
                    serializable_event[key] = value.isoformat()
                else:
                    serializable_event[key] = value
            serializable_events.append(serializable_event)
        
        with open(filename, 'w') as f:
            json.dump(serializable_events, f, indent=2)
        
        return filename


# Example usage
if __name__ == "__main__":
    import argparse
    from datetime import datetime, timedelta
    
    # Mock objects for testing
    class MockCityMap:
        def __init__(self):
            self.city_name = "Test City"
            self.zones = {
                "downtown": {
                    "name": "Downtown",
                    "center": (37.7749, -122.4194),
                    "radius": 2
                },
                "airport": {
                    "name": "Airport",
                    "center": (37.6213, -122.3790),
                    "radius": 2
                }
            }
        
        def generate_location(self, time, location_type, origin=None, destination=None):
            return {
                "latitude": 37.7749 + random.uniform(-0.1, 0.1),
                "longitude": -122.4194 + random.uniform(-0.1, 0.1),
                "address": f"{random.randint(1, 999)} Mock Street",
                "city": "Test City"
            }
        
        def calculate_distance(self, loc1, loc2):
            # Simple mock distance
            return random.uniform(1, 15)
    
    class MockDemandModel:
        def add_special_event(self, name, start_time, end_time, demand_multiplier, area_specific=False, area_name=None):
            print(f"Added demand event: {name}, multiplier: {demand_multiplier}")
    
    class MockTrafficModel:
        def add_special_event(self, name, start_time, end_time, traffic_multiplier):
            print(f"Added traffic event: {name}, multiplier: {traffic_multiplier}")
            
        def add_weather_condition(self, date, weather_type, intensity):
            print(f"Added weather on {date}: {weather_type}, intensity: {intensity}")
    
    class MockRideSimulator:
        def __init__(self):
            self.users = [{"user_id": f"U{i}", "platform": "iOS"} for i in range(10)]
            self.drivers = [{"driver_id": f"D{i}", "rating": 4.5} for i in range(5)]
    
    parser = argparse.ArgumentParser(description='Generate special events')
    parser.add_argument('--output', type=str, default='special_events.json', help='Output JSON file')
    parser.add_argument('--start_date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--duration_days', type=int, default=30, help='Duration in days (if not using start/end)')
    
    args = parser.parse_args()
    
    # Determine time range
    if args.start_date and args.end_date:
        start_date = datetime.fromisoformat(args.start_date)
        end_date = datetime.fromisoformat(args.end_date)
    else:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=args.duration_days)
    
    print(f"Generating special events from {start_date} to {end_date}")
    
    # Create generator with mock objects
    generator = SpecialEventsGenerator(
        MockCityMap(),
        MockDemandModel(),
        MockTrafficModel(),
        MockRideSimulator()
    )
    
    # Generate events
    events = generator.generate_special_events_for_period(start_date, end_date)
    
    # Save to file if requested
    if args.output:
        output_file = generator.save_to_json(args.output)
        print(f"Saved special events to {output_file}")
