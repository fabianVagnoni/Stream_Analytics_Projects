"""
Ride Simulator for Ride-Hailing Service

This module simulates the complete lifecycle of ride-hailing events,
generating realistic event sequences based on user and driver behavior patterns.

Usage:
    from ride_simulator import RideSimulator
    simulator = RideSimulator(users, drivers, city_map, demand_model, traffic_model)
    events = simulator.generate_rides(start_time, end_time)
"""

import random
import numpy as np
import uuid
from datetime import datetime, timedelta
import math
import json
import os

class RideSimulator:
    """
    Simulates ride-hailing events based on user and driver behavior models.
    """
    
    def __init__(self, users, drivers, city_map, demand_model, traffic_model):
        """
        Initialize the ride simulator with data models.
        
        Args:
            users: List of user data dictionaries
            drivers: List of driver data dictionaries
            city_map: CityMap object for geographic simulation
            demand_model: DemandModel object for temporal patterns
            traffic_model: TrafficModel object for traffic conditions
        """
        self.users = users
        self.drivers = drivers
        self.city_map = city_map
        self.demand_model = demand_model
        self.traffic_model = traffic_model
        self.active_drivers = {}
        self.payment_methods = ["CARD"] * 4 + ["CASH"]
        self.vehicle_pricing = {
            "Economy": 1.0,
            "Comfort": 1.5,
            "Premium": 2.2,
            "XL": 1.8
        }
        
        # App version and platform distributions
        self.app_versions = {
            "iOS": ["4.5.1", "4.5.2", "4.6.0", "4.6.1", "4.7.0"],
            "Android": ["4.5.0", "4.5.1", "4.6.0", "4.6.2", "4.7.0"],
            "Web": ["1.2.0", "1.2.1", "1.3.0"]
        }
        
        # App version weights (newer versions more common)
        self.version_weights = {
            "iOS": [0.1, 0.15, 0.2, 0.25, 0.3],
            "Android": [0.05, 0.1, 0.25, 0.25, 0.35],
            "Web": [0.2, 0.3, 0.5]
        }
    
    def calculate_fare(self, distance_km, vehicle_type, surge_multiplier):

        base_fare = 5.0 + distance_km * 1.2
        vehicle_price_factor = self.vehicle_pricing.get(vehicle_type, 1.0)
        total_fare = base_fare * vehicle_price_factor * surge_multiplier
        return round(total_fare, 2)

    def calculate_surge_multiplier(self, demand_factor, available_drivers):
        """Realistic surge pricing based on demand vs. available drivers."""
        if available_drivers == 0:
            return 3.0  # Max surge

        surge_multiplier = 1.0 + min(2.0, max(0, (demand_factor - 1) + (0.5 / available_drivers)))
        return round(surge_multiplier, 1)

    def calculate_user_rating(self, estimated_duration, actual_duration, surge_multiplier, traffic_descriptor, user, driver):
        """Realistic calculation of user rating."""
        rating = 5.0

        duration_ratio = actual_duration / estimated_duration
        if duration_ratio > 1.3:
            rating -= 1.0
        elif duration_ratio > 1.1:
            rating -= 0.5

        traffic_penalties = {"MEDIUM": -0.5, "HIGH": -1.0, "SEVERE": -1.5}
        rating += traffic_penalties.get(traffic_descriptor, 0)

        rating -= 0.5 * (surge_multiplier - 1)
        rating += (driver.get("rating", 4.5) - 4.5) / 2

        return round(max(1.0, min(5.0, rating)) * 2) / 2
        
    
    def generate_ride_events(self, user, start_time):
        """
        Generate a complete sequence of events for a single ride.
        
        Args:
            user: User data dictionary
            start_time: Datetime when the ride request is made
            
        Returns:
            list: Sequence of ride events
        """
        # Initialize event IDs and ride ID
        event_id_base = uuid.uuid4().hex[:12]
        ride_id = f"R-{uuid.uuid4().hex[:10]}"
        session_id = f"S-{uuid.uuid4().hex[:8]}"
        
        # Current timestamp for first event
        current_timestamp = int(start_time.timestamp() * 1000)
        
        events = []
        event_count = 1
        
        # Select app version based on platform
        platform = user.get("platform", random.choice(["iOS", "Android"]))
        app_version = random.choices(
            self.app_versions[platform],
            weights=self.version_weights[platform],
            k=1
        )[0]
        
        # 1. RIDE_REQUESTED event
        pickup_location = self.city_map.generate_location(start_time, "origin")
        dropoff_location = self.city_map.generate_location(
            start_time, 
            "destination", 
            origin=pickup_location
        )
        
        # Calculate ride details
        distance_km = self.city_map.calculate_distance(pickup_location, dropoff_location)
        
        # Apply some randomness to estimated duration (people are not perfect estimators)
        estimated_duration_minutes = int(distance_km * 2 + random.randint(5, 15))
        
        # Check time of day and location for surge pricing
        # More sophisticated surge pricing based on demand and supply
        hour = start_time.hour
        day_of_week = start_time.weekday()
        demand_factor = self.demand_model.get_demand_multiplier(start_time)
        


        # Calculate surge multiplier with adjusted demand factor
        surge_multiplier = self.calculate_surge_multiplier(demand_factor, len(self.drivers) - len(self.active_drivers))
        
        # Calculate fare
        vehicle_type = user.get("preferred_vehicle_type", random.choice(list(self.vehicle_pricing.keys())))
        total_fare = self.calculate_fare(distance_km, vehicle_type, surge_multiplier)
        
        # Generate traffic conditions based on time of day
        traffic_level = self.traffic_model.get_traffic_level(start_time)
        traffic_descriptor = self.traffic_model.get_traffic_descriptor(traffic_level)
        
        delay_minutes = 0
        if traffic_descriptor == "MEDIUM":
            delay_minutes = random.randint(5, 15)
        elif traffic_descriptor == "HIGH":
            delay_minutes = random.randint(15, 30)
        elif traffic_descriptor == "SEVERE":
            delay_minutes = random.randint(30, 60)
        
        traffic_conditions = {
            "traffic_level": traffic_descriptor,
            "estimated_delay_minutes": delay_minutes
        }
        
        # RIDE_REQUESTED event
        request_event = {
            "event_id": f"{event_id_base}_{event_count}",
            "ride_id": ride_id,
            "event_type": "RIDE_REQUESTED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": None,
            "pickup_location": pickup_location,
            "dropoff_location": dropoff_location,
            "ride_details": {
                "distance_km": distance_km,
                "estimated_duration_minutes": estimated_duration_minutes,
                "actual_duration_minutes": None,
                "vehicle_type": vehicle_type,
                "base_fare": round(self.calculate_fare(distance_km, vehicle_type, 1.0), 2),
                "surge_multiplier": surge_multiplier,
                "total_fare": round(total_fare, 2)
            },
            "payment_info": None,
            "ratings": None,
            "cancellation_info": None,
            "traffic_conditions": traffic_conditions,
            "driver_location": None,
            "app_version": app_version,
            "platform": platform,
            "session_id": session_id
        }
        
        events.append(request_event)
        event_count += 1
        
        # Check if user cancels the ride before driver assignment
        # Higher cancellation probability if surge is high
        user_cancel_probability = user.get("cancellation_rate", 0.05)
        user_cancel_probability *= (1 + (surge_multiplier - 1) * 0.5)  # More likely to cancel during surge
        
        if random.random() < user_cancel_probability:
            # User cancels before driver assigned
            current_timestamp += random.randint(10000, 120000)  # 10-120 seconds later
            
            cancel_event = {
                "event_id": f"{event_id_base}_{event_count}",
                "ride_id": ride_id,
                "event_type": "RIDE_CANCELED_BY_USER",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": None,
                "pickup_location": pickup_location,
                "dropoff_location": dropoff_location,
                "ride_details": request_event["ride_details"],
                "payment_info": None,
                "ratings": None,
                "cancellation_info": {
                    "canceled_by": "USER",
                    "cancellation_reason": random.choice([
                        "Changed plans", 
                        "Waiting too long",
                        "Requested by mistake",
                        "Found alternative transportation",
                        "Price too high"  # More likely during surge
                    ]),
                    "cancellation_fee": 0.0 if random.random() < 0.7 else round(total_fare * 0.25, 2)
                },
                "traffic_conditions": traffic_conditions,
                "driver_location": None,
                "app_version": app_version,
                "platform": platform,
                "session_id": session_id
            }
            
            events.append(cancel_event)
            return events
        
        # 2. DRIVER_ASSIGNED event
        # Find available driver
        driver = self._select_driver(pickup_location, vehicle_type)
        
        if not driver:
            # Rare case - no drivers available
            current_timestamp += random.randint(60000, 180000)  # 1-3 minutes later
            
            cancel_event = {
                "event_id": f"{event_id_base}_{event_count}",
                "ride_id": ride_id,
                "event_type": "RIDE_CANCELED_BY_SYSTEM",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": None,
                "pickup_location": pickup_location,
                "dropoff_location": dropoff_location,
                "ride_details": request_event["ride_details"],
                "payment_info": None,
                "ratings": None,
                "cancellation_info": {
                    "canceled_by": "SYSTEM",
                    "cancellation_reason": "No drivers available",
                    "cancellation_fee": 0.0
                },
                "traffic_conditions": traffic_conditions,
                "driver_location": None,
                "app_version": app_version,
                "platform": platform,
                "session_id": session_id
            }
            
            events.append(cancel_event)
            return events
        
        # Driver is assigned successfully
        current_timestamp += random.randint(5000, 30000)  # 5-30 seconds later
        
        # Generate initial driver location (some distance away from pickup)
        # Higher-rated drivers tend to be closer to pickup location
        max_distance = 0.05 * (5.0 - driver.get("rating", 4.0))  # 0.05 to 0.25 degrees
        driver_initial_location = {
            "latitude": pickup_location["latitude"] + random.uniform(-max_distance, max_distance),
            "longitude": pickup_location["longitude"] + random.uniform(-max_distance, max_distance),
            "heading": random.uniform(0, 359),
            "speed_kmh": random.uniform(15, 60)
        }
        
        driver_assigned_event = {
            "event_id": f"{event_id_base}_{event_count}",
            "ride_id": ride_id,
            "event_type": "DRIVER_ASSIGNED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": pickup_location,
            "dropoff_location": dropoff_location,
            "ride_details": request_event["ride_details"],
            "payment_info": None,
            "ratings": None,
            "cancellation_info": None,
            "traffic_conditions": traffic_conditions,
            "driver_location": driver_initial_location,
            "app_version": app_version,
            "platform": platform,
            "session_id": session_id
        }
        
        events.append(driver_assigned_event)
        event_count += 1
        
        # Mark this driver as busy
        self.active_drivers[driver["driver_id"]] = {
            "status": "assigned",
            "ride_id": ride_id,
            "location": driver_initial_location
        }
        
        # Check if driver cancels
        driver_cancel_probability = driver.get("cancellation_rate", 0.05)
        
        # Adjust based on surge and driver rating - lower rated drivers cancel more
        driver_rating_factor = (5.0 - driver.get("rating", 4.0)) * 0.1  # 0.1 to 0.5
        driver_cancel_probability += driver_rating_factor
        
        # Reduce cancellation probability during surge - drivers want those fares
        driver_cancel_probability *= max(0.5, 1.0 - (surge_multiplier - 1.0))
        
        if random.random() < driver_cancel_probability:
            # Driver cancels
            current_timestamp += random.randint(30000, 180000)  # 30sec-3min later
            
            cancel_event = {
                "event_id": f"{event_id_base}_{event_count}",
                "ride_id": ride_id,
                "event_type": "RIDE_CANCELED_BY_DRIVER",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": driver["driver_id"],
                "pickup_location": pickup_location,
                "dropoff_location": dropoff_location,
                "ride_details": request_event["ride_details"],
                "payment_info": None,
                "ratings": None,
                "cancellation_info": {
                    "canceled_by": "DRIVER",
                    "cancellation_reason": random.choice([
                        "Too far away",
                        "Vehicle issue",
                        "End of shift",
                        "Traffic conditions",
                        "Passenger unreachable"
                    ]),
                    "cancellation_fee": 0.0
                },
                "traffic_conditions": traffic_conditions,
                "driver_location": driver_initial_location,
                "app_version": app_version,
                "platform": platform,
                "session_id": session_id
            }
            
            events.append(cancel_event)
            
            # Mark driver as available again
            if driver["driver_id"] in self.active_drivers:
                del self.active_drivers[driver["driver_id"]]
                
            return events
        
        # 3. DRIVER_ARRIVED event
        # Calculate time to arrival based on distance to pickup
        distance_to_pickup = self.city_map.haversine_distance(
            driver_initial_location["latitude"], driver_initial_location["longitude"],
            pickup_location["latitude"], pickup_location["longitude"]
        )
        
        # Traffic affects driver arrival time
        delay_factor = self.traffic_model.get_trip_delay_factor(traffic_level)
        
        # Base speed around 30 km/h in city
        base_speed_kmh = 30
        actual_speed = base_speed_kmh / delay_factor
        
        # Calculate time in milliseconds
        travel_time_to_pickup = (distance_to_pickup / actual_speed) * 3600 * 1000
        
        # Add some randomness
        travel_time_to_pickup *= random.uniform(0.8, 1.2)
        
        # Update timestamp
        current_timestamp += int(travel_time_to_pickup)
        
        # Driver has arrived at pickup
        driver_arrived_event = {
            "event_id": f"{event_id_base}_{event_count}",
            "ride_id": ride_id,
            "event_type": "DRIVER_ARRIVED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": pickup_location,
            "dropoff_location": dropoff_location,
            "ride_details": request_event["ride_details"],
            "payment_info": None,
            "ratings": None,
            "cancellation_info": None,
            "traffic_conditions": traffic_conditions,
            "driver_location": {
                "latitude": pickup_location["latitude"],
                "longitude": pickup_location["longitude"],
                "heading": random.uniform(0, 359),
                "speed_kmh": 0.0
            },
            "app_version": app_version,
            "platform": platform,
            "session_id": session_id
        }
        
        events.append(driver_arrived_event)
        event_count += 1
        
        # Update driver status
        self.active_drivers[driver["driver_id"]]["status"] = "arrived"
        self.active_drivers[driver["driver_id"]]["location"] = {
            "latitude": pickup_location["latitude"],
            "longitude": pickup_location["longitude"],
            "heading": random.uniform(0, 359),
            "speed_kmh": 0.0
        }
        
        # 4. RIDE_STARTED event
        # Add waiting time at pickup (user gets in car)
        current_timestamp += random.randint(60000, 300000)  # 1-5 minutes
        
        ride_started_event = {
            "event_id": f"{event_id_base}_{event_count}",
            "ride_id": ride_id,
            "event_type": "RIDE_STARTED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": pickup_location,
            "dropoff_location": dropoff_location,
            "ride_details": request_event["ride_details"],
            "payment_info": None,
            "ratings": None,
            "cancellation_info": None,
            "traffic_conditions": traffic_conditions,
            "driver_location": {
                "latitude": pickup_location["latitude"],
                "longitude": pickup_location["longitude"],
                "heading": self._get_heading(pickup_location, dropoff_location),
                "speed_kmh": random.uniform(10, 30)
            },
            "app_version": app_version,
            "platform": platform,
            "session_id": session_id
        }
        
        events.append(ride_started_event)
        event_count += 1
        
        # Update driver status
        self.active_drivers[driver["driver_id"]]["status"] = "riding"
        
        # 5. RIDE_COMPLETED event
        # Calculate actual trip duration based on traffic and distance
        base_duration_minutes = distance_km * 2  # Approximate 30 km/h average speed
        actual_duration_minutes = int(base_duration_minutes * delay_factor)
        
        # Add some randomness to the actual duration
        actual_duration_minutes = int(actual_duration_minutes * random.uniform(0.9, 1.1))
        
        # Update the timestamp
        current_timestamp += actual_duration_minutes * 60 * 1000
        
        # Update ride details with actual duration
        ride_details = request_event["ride_details"].copy()
        ride_details["actual_duration_minutes"] = actual_duration_minutes
        
        ride_completed_event = {
            "event_id": f"{event_id_base}_{event_count}",
            "ride_id": ride_id,
            "event_type": "RIDE_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": pickup_location,
            "dropoff_location": dropoff_location,
            "ride_details": ride_details,
            "payment_info": None,
            "ratings": None,
            "cancellation_info": None,
            "traffic_conditions": traffic_conditions,
            "driver_location": {
                "latitude": dropoff_location["latitude"],
                "longitude": dropoff_location["longitude"],
                "heading": random.uniform(0, 359),
                "speed_kmh": 0.0
            },
            "app_version": app_version,
            "platform": platform,
            "session_id": session_id
        }
        
        events.append(ride_completed_event)
        event_count += 1
        
        # Update driver status and location
        self.active_drivers[driver["driver_id"]]["status"] = "completed"
        self.active_drivers[driver["driver_id"]]["location"] = {
            "latitude": dropoff_location["latitude"],
            "longitude": dropoff_location["longitude"],
            "heading": random.uniform(0, 359),
            "speed_kmh": 0.0
        }
        
        # 6. PAYMENT_COMPLETED event
        # Process payment right after ride completion
        current_timestamp += random.randint(5000, 30000)  # 5-30 seconds
        
        payment_method = random.choice(self.payment_methods)
        
        payment_info = {
            "payment_method": payment_method,
            "payment_status": "COMPLETED",
            "payment_id": f"P-{uuid.uuid4().hex[:10]}"
        }
        
        payment_completed_event = {
            "event_id": f"{event_id_base}_{event_count}",
            "ride_id": ride_id,
            "event_type": "PAYMENT_COMPLETED",
            "timestamp": current_timestamp,
            "user_id": user["user_id"],
            "driver_id": driver["driver_id"],
            "pickup_location": pickup_location,
            "dropoff_location": dropoff_location,
            "ride_details": ride_details,
            "payment_info": payment_info,
            "ratings": None,
            "cancellation_info": None,
            "traffic_conditions": traffic_conditions,
            "driver_location": {
                "latitude": dropoff_location["latitude"],
                "longitude": dropoff_location["longitude"],
                "heading": random.uniform(0, 359),
                "speed_kmh": 0.0
            },
            "app_version": app_version,
            "platform": platform,
            "session_id": session_id
        }
        
        events.append(payment_completed_event)
        event_count += 1
        
        # 7. USER_RATED_DRIVER event
        # Not all users rate immediately - some don't rate at all
        if random.random() < 0.7:  # 70% of users give ratings
            # Some users rate immediately, others later
            rating_delay = random.choice([
                random.randint(10000, 60000),  # 10sec-1min: Immediate
                random.randint(300000, 3600000),  # 5min-1hour: Short delay
                random.randint(3600000, 24*3600000)  # 1-24 hours: Long delay
            ])
            
            current_timestamp += rating_delay
            
            # Calculate user's rating based on various factors
            # Base rating distribution with 5 being most common
            rating_distribution = [1.0, 2.0, 3.0, 4.0, 5.0]
            rating_weights = [0.01, 0.04, 0.1, 0.25, 0.6]
            
            # Start with base rating tendency
            base_rating = random.choices(rating_distribution, weights=rating_weights, k=1)[0]
            
            # Factors that affect rating
            # 1. Actual vs Estimated duration
            duration_ratio = actual_duration_minutes / estimated_duration_minutes
            duration_factor = 0
            if duration_ratio > 1.3:  # Much longer than expected
                duration_factor = -1
            elif duration_ratio > 1.1:  # Longer than expected
                duration_factor = -0.5
            
            # 2. Traffic conditions
            traffic_factor = 0
            if traffic_descriptor == "HIGH":
                traffic_factor = -0.5
            elif traffic_descriptor == "SEVERE":
                traffic_factor = -1
            
            # 3. Surge pricing (higher surge might lead to more critical ratings)
            surge_factor = min(0, -0.5 * (surge_multiplier - 1.0))
            
            # 4. User's rating tendency
            user_rating_tendency = 0
            if "avg_rating_given" in user:
                # If user typically gives low ratings, adjust down
                if user["avg_rating_given"] < 4.0:
                    user_rating_tendency = (user["avg_rating_given"] - 4.0) / 2
            
            # 5. Driver's rating (good drivers tend to get good ratings)
            driver_factor = 0
            if "rating" in driver:
                driver_factor = (driver["rating"] - 4.0) / 2  # -0.5 to +0.5
            
            # Combine all factors
            rating_adjustment = duration_factor + traffic_factor + surge_factor + user_rating_tendency + driver_factor
            
            # Ensure the adjusted rating is within bounds
            rating = max(1.0, min(5.0, base_rating + rating_adjustment))
            
            # Generate a comment sometimes
            comment = None
            if random.random() < 0.3:  # 30% chance of comment
                if rating >= 4.5:
                    comment = random.choice([
                        "Great driver, very professional!",
                        "Excellent service!",
                        "Very friendly and skilled driver!",
                        "Clean car, smooth ride!",
                        "On time and efficient!",
                        "Perfect ride, thank you!"
                    ])
                elif rating >= 3.5:
                    comment = random.choice([
                        "Good ride overall.",
                        "Decent service.",
                        "Nice driver.",
                        "Got me there safely.",
                        "No issues."
                    ])
                else:
                    comment = random.choice([
                        "Driver was late.",
                        "Car was not very clean.",
                        "Driver took a longer route.",
                        "Driving was a bit rough.",
                        "Driver wasn't very friendly.",
                        "Not a good experience."
                    ])
            
            rating_info = {
                "user_to_driver_rating": rating,
                "driver_to_user_rating": None,
                "user_comment": comment,
                "driver_comment": None
            }
            
            user_rated_event = {
                "event_id": f"{event_id_base}_{event_count}",
                "ride_id": ride_id,
                "event_type": "USER_RATED_DRIVER",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": driver["driver_id"],
                "pickup_location": pickup_location,
                "dropoff_location": dropoff_location,
                "ride_details": ride_details,
                "payment_info": payment_info,
                "ratings": rating_info,
                "cancellation_info": None,
                "traffic_conditions": traffic_conditions,
                "driver_location": None,
                "app_version": app_version,
                "platform": platform,
                "session_id": session_id
            }
            
            events.append(user_rated_event)
            event_count += 1
        
        # 8. DRIVER_RATED_USER event
        # Drivers tend to rate more consistently
        if random.random() < 0.9:  # 90% of drivers give ratings
            # Drivers usually rate soon after the ride
            driver_rating_delay = random.randint(10000, 900000)  # 10sec-15min
            
            # But make sure it happens after the user rating if there was one
            if events[-1]["event_type"] == "USER_RATED_DRIVER":
                current_timestamp = max(current_timestamp, events[-1]["timestamp"] + 5000)
            else:
                current_timestamp += driver_rating_delay
            
            # Driver ratings tend to be higher on average
            driver_rating_distribution = [1.0, 2.0, 3.0, 4.0, 5.0]
            driver_rating_weights = [0.01, 0.02, 0.07, 0.2, 0.7]
            
            driver_rating = random.choices(driver_rating_distribution, weights=driver_rating_weights, k=1)[0]
            
            # Adjust based on user's profile
            # If the user has a high cancellation rate, they may get lower ratings
            if "cancellation_rate" in user and user["cancellation_rate"] > 0.1:
                driver_rating -= user["cancellation_rate"] * 10  # Up to -1 point
            
            # If payment was cash, slightly lower rating on average
            if payment_info["payment_method"] == "CASH":
                driver_rating -= 0.5 if random.random() < 0.3 else 0
            
            # Ensure rating is within bounds
            driver_rating = max(1.0, min(5.0, driver_rating))
            
            # Generate comment occasionally
            driver_comment = None
            if random.random() < 0.2:  # 20% chance
                if driver_rating >= 4.5:
                    driver_comment = random.choice([
                        "Great passenger!",
                        "Very polite and on time!",
                        "Perfect rider!",
                        "Respectful and friendly!",
                        "Excellent communication!",
                    ])
                elif driver_rating >= 3.5:
                    driver_comment = random.choice([
                        "Good passenger.",
                        "No issues.",
                        "OK ride.",
                        "Decent customer."
                    ])
                else:
                    driver_comment = random.choice([
                        "Made me wait.",
                        "Not very friendly.",
                        "Communication issues.",
                        "Left trash in car."
                    ])
            
            # Update ratings info
            rating_info = {}
            if events[-1]["event_type"] == "USER_RATED_DRIVER":
                # Copy existing user rating
                rating_info = events[-1]["ratings"].copy()
            else:
                rating_info = {
                    "user_to_driver_rating": None,
                    "user_comment": None
                }
            
            rating_info["driver_to_user_rating"] = driver_rating
            rating_info["driver_comment"] = driver_comment
            
            driver_rated_event = {
                "event_id": f"{event_id_base}_{event_count}",
                "ride_id": ride_id,
                "event_type": "DRIVER_RATED_USER",
                "timestamp": current_timestamp,
                "user_id": user["user_id"],
                "driver_id": driver["driver_id"],
                "pickup_location": pickup_location,
                "dropoff_location": dropoff_location,
                "ride_details": ride_details,
                "payment_info": payment_info,
                "ratings": rating_info,
                "cancellation_info": None,
                "traffic_conditions": traffic_conditions,
                "driver_location": None,
                "app_version": app_version,
                "platform": platform,
                "session_id": session_id
            }
            
            events.append(driver_rated_event)
        
        # Mark driver as available again
        if driver["driver_id"] in self.active_drivers:
            del self.active_drivers[driver["driver_id"]]
        
        # After ride completion, calculate user rating
        actual_duration_minutes = ride_details["actual_duration_minutes"]  # Assuming this is calculated earlier
        traffic_descriptor = self.traffic_model.get_traffic_descriptor(traffic_level)
        user_rating = self.calculate_user_rating(estimated_duration_minutes, actual_duration_minutes, surge_multiplier, traffic_descriptor, user, driver)

        # Update ratings in the event
        ride_completed_event["ratings"] = {"user_to_driver_rating": user_rating}
        
        return events
    
    def generate_rides(self, start_time, end_time, output_file=None, batch_size=1000):
        """
        Generate rides for a time period.
        
        Args:
            start_time: Datetime to start generating from
            end_time: Datetime to end generation
            output_file: File to save events to (optional)
            batch_size: Number of events to save at once (if output_file is provided)
            
        Returns:
            list: All generated events if output_file is None
        """
        all_events = []
        event_count = 0
        current_time = start_time
        
        # Open output file if specified
        if output_file:
            out_file = open(output_file, 'w')
            out_file.write('[\n')  # Start JSON array
            first_event = True
        
        print(f"Generating rides from {start_time} to {end_time}")
        
        while current_time < end_time:
            # Calculate expected number of ride requests for this hour
            demand_multiplier = self.demand_model.get_demand_multiplier(current_time)
            expected_rides = max(1, int(self.demand_model.base_demand * demand_multiplier))
            
            # Add some randomness
            actual_rides = int(np.random.poisson(expected_rides))
            
            print(f"Time: {current_time}, Demand Multiplier: {demand_multiplier:.2f}, Generating {actual_rides} rides")
            
            # Generate ride requests distributed throughout the hour
            for i in range(actual_rides):
                # Distribute requests within the hour
                minutes_offset = random.random() * 60
                request_time = current_time + timedelta(minutes=minutes_offset)
                
                if request_time >= end_time:
                    break
                
                # Select a user likely to request a ride at this time
                user = self._select_user_for_time(request_time)
                
                # Generate complete ride event sequence
                events = self.generate_ride_events(user, request_time)
                
                # Save or collect events
                if output_file:
                    for event in events:
                        if not first_event:
                            out_file.write(',\n')
                        else:
                            first_event = False
                        
                        json.dump(event, out_file)
                        event_count += 1
                        
                        # Flush periodically
                        if event_count % batch_size == 0:
                            out_file.flush()
                else:
                    all_events.extend(events)
            
            # Move to the next hour
            current_time += timedelta(hours=1)
        
        # Close output file if used
        if output_file:
            out_file.write('\n]')  # End JSON array
            out_file.close()
            return event_count
        else:
            return all_events
    
    def _select_user_for_time(self, current_time):
        """
        Select a user who is likely to request a ride at the given time.
        
        Args:
            current_time: Time of the ride request
            
        Returns:
            dict: Selected user data
        """
        # Weigh users based on their likelihood to use the service at this time
        hour = current_time.hour
        day_of_week = current_time.weekday()
        is_weekend = day_of_week >= 5
        
        weights = []
        
        for user in self.users:
            # Base weight is proportional to rides_per_month if available
            base_weight = user.get("rides_per_month", 10) / 10
            
            # Apply archetype-specific time weights
            archetype = user.get("archetype", "occasional")
            
            if archetype == "Daily commuter":
                if is_weekend:
                    weight = base_weight * 0.2  # Low weekend usage
                elif 7 <= hour <= 9 or 16 <= hour <= 19:
                    weight = base_weight * 2.0  # High usage during commute
                else:
                    weight = base_weight * 0.5  # Lower usage other times
            
            elif archetype == "Business traveler":
                if is_weekend:
                    weight = base_weight * 0.3  # Low weekend usage
                elif 6 <= hour <= 22:  # Active during business day and evening
                    weight = base_weight * 1.5
                else:
                    weight = base_weight * 0.2
            
            elif archetype == "Weekend socialite":
                if is_weekend:
                    if 11 <= hour <= 14 or 19 <= hour <= 23:
                        weight = base_weight * 2.0  # High weekend social hours
                    else:
                        weight = base_weight * 1.0  # Regular weekend usage
                elif 19 <= hour <= 23:  # Weekday evenings
                    weight = base_weight * 1.0
                else:
                    weight = base_weight * 0.3
            
            else:  # "Occasional" or unknown
                # More random pattern
                weight = base_weight * random.uniform(0.5, 1.5)
            
            weights.append(max(0.1, weight))
        
        # Select user based on weights
        selected_user = random.choices(self.users, weights=weights, k=1)[0]
        return selected_user
    
    def _select_driver(self, pickup_location, vehicle_type):
        """
        Select an available driver near the pickup location.
        
        Args:
            pickup_location: Location dictionary
            vehicle_type: Type of vehicle requested
            
        Returns:
            dict: Selected driver data or None if no suitable driver found
        """
        # First, filter active drivers (already in a ride)
        available_drivers = [d for d in self.drivers if d["driver_id"] not in self.active_drivers]
        
        # Filter by vehicle type if needed
        if vehicle_type != "Economy":
            # For non-economy rides, find matching vehicle type or higher
            vehicle_hierarchy = ["Economy", "Comfort", "Premium", "XL"]
            min_level = vehicle_hierarchy.index(vehicle_type)
            
            # Filter drivers with appropriate vehicle type
            qualified_drivers = []
            for driver in available_drivers:
                vehicle_desc = driver.get("vehicle", "").lower()
                
                # Determine vehicle type from description
                if "bmw" in vehicle_desc or "mercedes" in vehicle_desc or "audi" in vehicle_desc or "lexus" in vehicle_desc:
                    driver_type = "Premium"
                elif "sienna" in vehicle_desc or "odyssey" in vehicle_desc or "suburban" in vehicle_desc or "explorer" in vehicle_desc:
                    driver_type = "XL"
                elif "camry" in vehicle_desc or "accord" in vehicle_desc or "passat" in vehicle_desc or "mazda" in vehicle_desc:
                    driver_type = "Comfort"
                else:
                    driver_type = "Economy"
                
                # Check if driver's vehicle is adequate
                if vehicle_hierarchy.index(driver_type) >= min_level:
                    qualified_drivers.append(driver)
            
            available_drivers = qualified_drivers if qualified_drivers else available_drivers
        
        # If no drivers available, return None
        if not available_drivers:
            return None
        
        # Simulate driver assignment based on proximity and rating
        # Higher rated drivers are more likely to be assigned
        weights = []
        
        for driver in available_drivers:
            # Base weight is higher for better-rated drivers
            rating = driver.get("rating", 4.0)
            base_weight = (rating - 3.0) ** 2  # Exponential preference for high ratings
            
            # Simulate proximity by using driver ID as a seed
            # This creates a consistent but varied "location" for each driver
            driver_id_seed = int(driver["driver_id"].replace("D", ""))
            random.seed(driver_id_seed)
            
            # Generate a simulated distance
            # Lower value means driver is closer to pickup
            simulated_distance = random.uniform(0.5, 10.0)
            
            # Reset random seed
            random.seed()
            
            # Closer drivers get higher weights
            proximity_weight = 10.0 / (simulated_distance + 1.0)
            
            # Combine factors - rating and proximity
            weight = base_weight * proximity_weight
            
            weights.append(weight)
        
        # Select driver based on weights
        selected_driver = random.choices(available_drivers, weights=weights, k=1)[0]
        return selected_driver
    
    @staticmethod
    def _get_heading(start, end):
        """Calculate heading from start to end location in degrees"""
        # Convert to radians
        lat1 = math.radians(start["latitude"])
        lon1 = math.radians(start["longitude"])
        lat2 = math.radians(end["latitude"])
        lon2 = math.radians(end["longitude"])
        
        # Calculate heading
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        heading = math.degrees(math.atan2(y, x))
        
        # Normalize to 0-360
        heading = (heading + 360) % 360
        
        return heading



# Example usage
if __name__ == "__main__":
    import argparse
    import os
    from datetime import datetime, timedelta
    
    # This would normally import other modules, but for simplicity we'll mock them here
    class MockCityMap:
        def generate_location(self, time, location_type, origin=None, destination=None):
            return {
                "latitude": 37.7749 + random.uniform(-0.1, 0.1),
                "longitude": -122.4194 + random.uniform(-0.1, 0.1),
                "address": f"{random.randint(1, 999)} Mock Street",
                "city": "Madrid"
            }
        
        def calculate_distance(self, loc1, loc2):
            return random.uniform(1, 15)
        
        def haversine_distance(self, lat1, lon1, lat2, lon2):
            return random.uniform(1, 10)
    
    class MockDemandModel:
        def __init__(self):
            self.base_demand = 50
            
        def get_demand_multiplier(self, time):
            # Peak hours have higher demand
            hour = time.hour
            if 7 <= hour <= 9 or 16 <= hour <= 19:
                return random.uniform(1.5, 2.5)
            elif 0 <= hour < 6:
                return random.uniform(0.1, 0.5)
            else:
                return random.uniform(0.7, 1.3)
    
    class MockTrafficModel:
        def get_traffic_level(self, time):
            # Peak hours have higher traffic
            hour = time.hour
            if 7 <= hour <= 9 or 16 <= hour <= 19:
                return random.uniform(0.6, 0.9)
            elif 0 <= hour < 6:
                return random.uniform(0.1, 0.3)
            else:
                return random.uniform(0.3, 0.6)
        
        def get_traffic_descriptor(self, level):
            if level < 0.3:
                return "LOW"
            elif level < 0.6:
                return "MEDIUM"
            elif level < 0.8:
                return "HIGH"
            else:
                return "SEVERE"
        
        def get_trip_delay_factor(self, level):
            return 1 + level
    
    parser = argparse.ArgumentParser(description='Generate ride simulation data')
    parser.add_argument('--output', type=str, default='data/ride_events.json', help='Output JSON file')
    parser.add_argument('--start_date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--duration_hours', type=int, default=24, help='Duration in hours (if not using start/end)')
    parser.add_argument('--users', type=int, default=50, help='Number of mock users to generate')
    parser.add_argument('--drivers', type=int, default=20, help='Number of mock drivers to generate')
    
    args = parser.parse_args()
    
    # Determine time range
    if args.start_date and args.end_date:
        start_time = datetime.fromisoformat(args.start_date)
        end_time = datetime.fromisoformat(args.end_date)
    else:
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=args.duration_hours)
    
    print(f"Generating rides from {start_time} to {end_time}")
    
    # Create mock users and drivers for testing
    mock_users = []
    for i in range(args.users):
        user_id = f"U{str(i).zfill(6)}"
        user = {
            "user_id": user_id,
            "rides_per_month": random.uniform(1, 30),
            "cancellation_rate": random.uniform(0.01, 0.2),
            "archetype": random.choice(["Daily commuter", "Business traveler", "Weekend socialite", "Occasional user"]),
            "platform": random.choice(["iOS", "Android", "Web"]),
            "preferred_vehicle_type": random.choice(["Economy", "Economy", "Economy", "Comfort", "Premium", "XL"])
        }
        mock_users.append(user)
    
    mock_drivers = []
    for i in range(args.drivers):
        driver_id = f"D{str(i).zfill(6)}"
        driver = {
            "driver_id": driver_id,
            "rating": random.uniform(3.5, 5.0),
            "cancellation_rate": random.uniform(0.01, 0.15),
            "vehicle": random.choice([
                "Toyota Corolla 2020",
                "Honda Civic 2019",
                "BMW 3 Series 2021",
                "Mercedes C-Class 2020",
                "Toyota Sienna 2018",
                "Hyundai Elantra 2019",
                "Mazda 6 2020"
            ])
        }
        mock_drivers.append(driver)
    
    # Create simulator
    simulator = RideSimulator(
        mock_users,
        mock_drivers,
        MockCityMap(),
        MockDemandModel(),
        MockTrafficModel()
    )
    
    # Generate and save rides
    event_count = simulator.generate_rides(start_time, end_time, args.output)
    
    print(f"Generated {event_count} events, saved to {args.output}")
