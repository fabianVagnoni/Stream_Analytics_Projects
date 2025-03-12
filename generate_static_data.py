"""
Static Data Generator for Ride-Hailing Simulation

This module generates static user and driver data based on persona archetypes,
creating realistic distributions of driver types, experience levels, and user behaviors.

Usage:
    python generate_static_data.py --drivers 500 --users 1000 --output_dir ./data
"""

import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import uuid
import json
import os
import argparse

# Initialize faker
fake = Faker()

def generate_drivers(count=500, city="San Francisco"):
    """
    Generate driver static and dynamic data based on driver personas.
    
    Args:
        count: Number of drivers to generate
        city: Primary city for the drivers
        
    Returns:
        tuple: (drivers_static, drivers_dynamic) lists containing driver data
    """
    drivers_static = []
    drivers_dynamic = []
    
    # Define driver archetypes with distinct behavioral patterns
    archetypes = [
        {"name": "Full-time pro", "weight": 0.3, "hours_per_week": (35, 50), "experience_years": (1, 5),
         "cancellation_rate": (0.02, 0.05), "rating": (4.2, 4.9)},
        {"name": "Part-time evening", "weight": 0.25, "hours_per_week": (15, 25), "experience_years": (0.5, 3),
         "cancellation_rate": (0.05, 0.1), "rating": (4.0, 4.8)},
        {"name": "Weekend warrior", "weight": 0.2, "hours_per_week": (8, 16), "experience_years": (0.2, 2),
         "cancellation_rate": (0.08, 0.15), "rating": (3.9, 4.7)},
        {"name": "Newbie", "weight": 0.15, "hours_per_week": (10, 30), "experience_years": (0, 0.5),
         "cancellation_rate": (0.1, 0.2), "rating": (3.5, 4.5)},
        {"name": "Inconsistent", "weight": 0.1, "hours_per_week": (5, 20), "experience_years": (0.1, 4),
         "cancellation_rate": (0.15, 0.25), "rating": (3.2, 4.6)},
    ]
    
    # Vehicle types with realistic distributions
    vehicle_types = [
        ("Economy", 0.6, ["Toyota Corolla", "Honda Civic", "Hyundai Elantra", "Nissan Sentra", "Kia Forte"]),
        ("Comfort", 0.25, ["Mazda 6", "Volkswagen Passat", "Subaru Legacy", "Honda Accord", "Toyota Camry"]),
        ("Premium", 0.1, ["BMW 3 Series", "Mercedes C-Class", "Audi A4", "Lexus ES", "Volvo S60"]),
        ("XL", 0.05, ["Toyota Sienna", "Honda Odyssey", "Ford Explorer", "Chevrolet Suburban", "GMC Yukon"])
    ]
    
    archetype_weights = [a["weight"] for a in archetypes]
    
    # Generate drivers with different characteristics
    for i in range(count):
        driver_id = f"D{str(i).zfill(6)}"
        
        # Select driver archetype
        archetype = random.choices(archetypes, weights=archetype_weights, k=1)[0]
        
        # Generate static data
        first_name = fake.first_name()
        last_name = fake.last_name()
        
        # Experience affects creation date
        experience_years = random.uniform(*archetype["experience_years"])
        days_since_signup = int(experience_years * 365)
        account_creation_date = (datetime.now() - timedelta(days=days_since_signup)).strftime("%Y-%m-%d")
        
        # Select vehicle type and specific model
        vehicle_type, _, models = random.choices(vehicle_types, 
                                            weights=[vt[1] for vt in vehicle_types], 
                                            k=1)[0]
        vehicle_year = 2017 + random.randint(0, 6)
        vehicle_model = random.choice(models)
        vehicle = f"{vehicle_model} {vehicle_year}"
        
        # Static driver data
        driver_static = {
            "driver_id": driver_id,
            "first_name": first_name,
            "last_name": last_name,
            "phone_number": fake.phone_number(),
            "email": f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 999)}@{fake.domain_name()}",
            "license_number": f"{fake.state_abbr()}{random.randint(10000, 99999)}",
            "vehicle": vehicle,
            "account_creation_date": account_creation_date
        }
        
        # Generate dynamic data based on driver archetype
        # Add variation with normal distribution around the archetype's values
        rating = min(5.0, max(1.0, np.random.normal(
            (archetype["rating"][0] + archetype["rating"][1]) / 2, 
            (archetype["rating"][1] - archetype["rating"][0]) / 4
        )))
        
        # Round to 1 decimal place
        rating = round(rating, 1)
        
        hours_per_week = random.uniform(*archetype["hours_per_week"])
        
        # Calculate rides based on experience and hours worked
        # More experienced drivers are more efficient
        efficiency_factor = 1.0 + (experience_years / 5) * 0.5  # Up to 50% more efficient with 5 years experience
        rides_factor = experience_years * hours_per_week * efficiency_factor * (0.8 + 0.4 * random.random())
        no_of_rides = int(rides_factor * 20)  # Scale factor to get realistic number
        
        # Money earned correlates with rides and rating
        avg_fare = 15 + (rating - 4) * 10  # Higher-rated drivers earn more per ride
        money_earned = no_of_rides * avg_fare * (0.9 + 0.2 * random.random())
        
        # Add some randomized variation to cancellation rate based on archetype
        cancellation_rate = min(1.0, max(0.01, np.random.normal(
            (archetype["cancellation_rate"][0] + archetype["cancellation_rate"][1]) / 2,
            (archetype["cancellation_rate"][1] - archetype["cancellation_rate"][0]) / 4
        )))
        
        # Round to 3 decimal places
        cancellation_rate = round(cancellation_rate, 3)
        
        # Dynamic driver data
        driver_dynamic = {
            "driver_id": driver_id,
            "rating": rating,
            "no_of_rides": no_of_rides,
            "cancellation_rate": cancellation_rate,
            "money_earned": round(money_earned, 2),
            "time_driven": round(hours_per_week * experience_years * 52, 1),  # Total hours
            "last_updated": int(datetime.now().timestamp() * 1000)
        }
        
        drivers_static.append(driver_static)
        drivers_dynamic.append(driver_dynamic)
    
    return drivers_static, drivers_dynamic

def generate_users(count=1000, city="San Francisco"):
    """
    Generate user static and dynamic data based on user personas.
    
    Args:
        count: Number of users to generate
        city: Primary city for the users
        
    Returns:
        tuple: (users_static, users_dynamic) lists containing user data
    """
    users_static = []
    users_dynamic = []
    
    # Define user archetypes with distinct behavioral patterns
    archetypes = [
        {"name": "Daily commuter", "weight": 0.4, "rides_per_month": (40, 60), 
         "cancellation_rate": (0.01, 0.05), "avg_rating_given": (4.0, 4.8),
         "experience_years": (0.5, 3)},
        {"name": "Business traveler", "weight": 0.15, "rides_per_month": (15, 30),
         "cancellation_rate": (0.05, 0.1), "avg_rating_given": (3.8, 4.5),
         "experience_years": (1, 4)},
        {"name": "Weekend socialite", "weight": 0.25, "rides_per_month": (10, 20),
         "cancellation_rate": (0.05, 0.15), "avg_rating_given": (3.5, 4.6),
         "experience_years": (0.5, 2)},
        {"name": "Occasional user", "weight": 0.2, "rides_per_month": (2, 8),
         "cancellation_rate": (0.1, 0.2), "avg_rating_given": (3.2, 4.9),
         "experience_years": (0.1, 5)}
    ]
    
    # Platform distribution
    platforms = [
        ("iOS", 0.55),
        ("Android", 0.44),
        ("Web", 0.01)
    ]
    
    archetype_weights = [a["weight"] for a in archetypes]
    platform_options = [p[0] for p in platforms]
    platform_weights = [p[1] for p in platforms]
    
    # Generate users with different characteristics
    for i in range(count):
        user_id = f"U{str(i).zfill(6)}"
        
        # Select user archetype
        archetype = random.choices(archetypes, weights=archetype_weights, k=1)[0]
        
        # Generate static data
        first_name = fake.first_name()
        last_name = fake.last_name()
        
        # Experience affects signup date
        experience_years = random.uniform(*archetype["experience_years"])
        days_since_signup = int(experience_years * 365)
        signup_date = (datetime.now() - timedelta(days=days_since_signup)).strftime("%Y-%m-%d")
        
        # Select platform
        platform = random.choices(platform_options, weights=platform_weights, k=1)[0]
        
        # Static user data
        user_static = {
            "user_id": user_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 999)}@{fake.domain_name()}",
            "phone_number": fake.phone_number(),
            "signup_date": signup_date,
            "city": city
        }
        
        # Dynamic user data based on archetype
        rides_per_month = random.uniform(*archetype["rides_per_month"])
        total_months = experience_years * 12
        rides_taken = int(rides_per_month * total_months)
        
        # Add some randomized variation to ratings and cancellation
        avg_rating = min(5.0, max(1.0, np.random.normal(
            (archetype["avg_rating_given"][0] + archetype["avg_rating_given"][1]) / 2,
            (archetype["avg_rating_given"][1] - archetype["avg_rating_given"][0]) / 4
        )))
        
        cancellation_rate = min(1.0, max(0.0, np.random.normal(
            (archetype["cancellation_rate"][0] + archetype["cancellation_rate"][1]) / 2,
            (archetype["cancellation_rate"][1] - archetype["cancellation_rate"][0]) / 4
        )))
        
        # Calculate money spent based on rides taken
        avg_fare = 15 + random.uniform(-5, 15)  # Base fare with variation
        money_spent = rides_taken * avg_fare * (0.9 + 0.2 * random.random())
        
        # Calculate last ride date (more frequent users have more recent rides)
        max_days_since_last_ride = max(1, int(30 / rides_per_month * 30))  # More frequent = more recent
        days_since_last_ride = min(days_since_signup, int(random.gammavariate(1, max_days_since_last_ride / 3)))
        last_ride_date = (datetime.now() - timedelta(days=days_since_last_ride)).strftime("%Y-%m-%d")
        
        user_dynamic = {
            "user_id": user_id,
            "rides_taken": rides_taken,
            "money_spent": round(money_spent, 2),
            "avg_rating_given": round(avg_rating, 1),
            "cancellation_rate": round(cancellation_rate, 3),
            "last_ride_date": last_ride_date,
            "last_updated": int(datetime.now().timestamp() * 1000)
        }
        
        # Add extra fields for simulation use (not in AVRO schema)
        user_dynamic["platform"] = platform
        user_dynamic["archetype"] = archetype["name"]
        user_dynamic["rides_per_month"] = rides_per_month
        
        users_static.append(user_static)
        users_dynamic.append(user_dynamic)
    
    return users_static, users_dynamic

def save_to_json(data, filename):
    """Save data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    """Main function to generate and save data"""
    parser = argparse.ArgumentParser(description='Generate static data for ride-hailing simulation')
    parser.add_argument('--drivers', type=int, default=500, help='Number of drivers to generate')
    parser.add_argument('--users', type=int, default=1000, help='Number of users to generate')
    parser.add_argument('--city', type=str, default='San Francisco', help='Primary city')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory for data files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate driver data
    print(f"Generating {args.drivers} drivers...")
    drivers_static, drivers_dynamic = generate_drivers(args.drivers, args.city)
    
    # Generate user data
    print(f"Generating {args.users} users...")
    users_static, users_dynamic = generate_users(args.users, args.city)
    
    # Save to JSON files
    print("Saving data to JSON files...")
    save_to_json(drivers_static, os.path.join(args.output_dir, 'drivers_static.json'))
    save_to_json(drivers_dynamic, os.path.join(args.output_dir, 'drivers_dynamic.json'))
    save_to_json(users_static, os.path.join(args.output_dir, 'users_static.json'))
    save_to_json(users_dynamic, os.path.join(args.output_dir, 'users_dynamic.json'))
    
    print("Data generation complete!")
    print(f"Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()
