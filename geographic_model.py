"""
Geographic Model for Ride-Hailing Simulation

This module creates a virtual city map with different zones and their characteristics,
providing functions to generate realistic locations and calculate distances between points.

Usage:
    from geographic_model import CityMap
    city = CityMap("San Francisco")
    pickup_location = city.generate_location(datetime.now(), "origin")
"""

import random
import math
import numpy as np
from datetime import datetime, time
import json

class CityMap:
    """
    Represents a virtual city map with different zones and their characteristics.
    """
    
    def __init__(self, city_name="San Francisco"):
        """
        Initialize the city map with predefined zones.
        
        Args:
            city_name: Name of the city for reference
        """
        self.city_name = city_name
        self.zones = self._create_zones()
        
        # Set city bounds based on zone locations
        self.lat_min = min([z["center"][0] - z["radius"]/111 for z in self.zones.values()])
        self.lat_max = max([z["center"][0] + z["radius"]/111 for z in self.zones.values()])
        self.lng_min = min([z["center"][1] - z["radius"]/111 for z in self.zones.values()])
        self.lng_max = max([z["center"][1] + z["radius"]/111 for z in self.zones.values()])
    
    def _create_zones(self):
        """Create a virtual city with different zones and characteristics"""
        if self.city_name == "San Francisco":
            return {
                "downtown": {
                    "name": "Downtown / Financial District",
                    "center": (37.7749, -122.4194),
                    "radius": 2,  # km
                    "popularity": 0.35,  # 35% of rides start/end here
                    "business_coefficient": 0.8,  # Higher during business hours
                    "nightlife_coefficient": 0.4,  # Moderate during night hours
                    "residential_coefficient": 0.2,  # Low residential density
                    "address_templates": [
                        "{} Market Street",
                        "{} Montgomery Street",
                        "{} California Street",
                        "{} Mission Street",
                        "{} Bush Street"
                    ]
                },
                "airport": {
                    "name": "SFO Airport",
                    "center": (37.6213, -122.3790),
                    "radius": 2,
                    "popularity": 0.15,
                    "business_coefficient": 0.4,
                    "nightlife_coefficient": 0.2,
                    "residential_coefficient": 0.0,
                    "address_templates": [
                        "SFO Terminal {}",
                        "San Francisco International Airport, Terminal {}",
                        "Airport Boulevard, Terminal {}"
                    ]
                },
                "marina": {
                    "name": "Marina District",
                    "center": (37.8030, -122.4370),
                    "radius": 1.5,
                    "popularity": 0.10,
                    "business_coefficient": 0.3,
                    "nightlife_coefficient": 0.6,
                    "residential_coefficient": 0.6,
                    "address_templates": [
                        "{} Chestnut Street",
                        "{} Lombard Street",
                        "{} Marina Boulevard",
                        "{} Fillmore Street",
                        "{} Union Street"
                    ]
                },
                "mission": {
                    "name": "Mission District",
                    "center": (37.7599, -122.4148),
                    "radius": 1.5,
                    "popularity": 0.15,
                    "business_coefficient": 0.4,
                    "nightlife_coefficient": 0.8,
                    "residential_coefficient": 0.6,
                    "address_templates": [
                        "{} Valencia Street",
                        "{} Mission Street",
                        "{} 16th Street",
                        "{} 24th Street",
                        "{} Dolores Street"
                    ]
                },
                "soma": {
                    "name": "SoMa",
                    "center": (37.7785, -122.4056),
                    "radius": 1.5,
                    "popularity": 0.15,
                    "business_coefficient": 0.7,
                    "nightlife_coefficient": 0.7,
                    "residential_coefficient": 0.4,
                    "address_templates": [
                        "{} Folsom Street",
                        "{} Harrison Street",
                        "{} Townsend Street",
                        "{} Bryant Street",
                        "{} 3rd Street"
                    ]
                },
                "richmond": {
                    "name": "Richmond District",
                    "center": (37.7816, -122.4836),
                    "radius": 2,
                    "popularity": 0.05,
                    "business_coefficient": 0.2,
                    "nightlife_coefficient": 0.2,
                    "residential_coefficient": 0.8,
                    "address_templates": [
                        "{} Geary Boulevard",
                        "{} Clement Street",
                        "{} Fulton Street",
                        "{} California Street",
                        "{} Park Presidio Boulevard"
                    ]
                },
                "sunset": {
                    "name": "Sunset District",
                    "center": (37.7504, -122.4900),
                    "radius": 2,
                    "popularity": 0.05,
                    "business_coefficient": 0.2,
                    "nightlife_coefficient": 0.1,
                    "residential_coefficient": 0.8,
                    "address_templates": [
                        "{} Irving Street",
                        "{} Judah Street",
                        "{} Taraval Street",
                        "{} Noriega Street",
                        "{} Sunset Boulevard"
                    ]
                }
            }
        else:
            # Default generic city if not San Francisco
            return {
                "downtown": {
                    "name": "Downtown",
                    "center": (40.7128, -74.0060),  # NYC coordinates as default
                    "radius": 3,
                    "popularity": 0.3,
                    "business_coefficient": 0.7,
                    "nightlife_coefficient": 0.6,
                    "residential_coefficient": 0.3,
                    "address_templates": [
                        "{} Main Street",
                        "{} Broadway",
                        "{} 1st Avenue",
                        "{} Center Street",
                        "{} Market Street"
                    ]
                },
                "airport": {
                    "name": "Airport",
                    "center": (40.6413, -73.7781),
                    "radius": 2,
                    "popularity": 0.15,
                    "business_coefficient": 0.3,
                    "nightlife_coefficient": 0.1,
                    "residential_coefficient": 0.0,
                    "address_templates": [
                        "Airport Terminal {}",
                        "Terminal {} Departures",
                        "Terminal {} Arrivals"
                    ]
                },
                "residential_north": {
                    "name": "North Residential",
                    "center": (40.7500, -74.0100),
                    "radius": 2.5,
                    "popularity": 0.2,
                    "business_coefficient": 0.1,
                    "nightlife_coefficient": 0.2,
                    "residential_coefficient": 0.8,
                    "address_templates": [
                        "{} Oak Street",
                        "{} Pine Street",
                        "{} Cedar Avenue",
                        "{} Maple Drive",
                        "{} Park Avenue"
                    ]
                },
                "shopping_district": {
                    "name": "Shopping District",
                    "center": (40.7300, -74.0000),
                    "radius": 2,
                    "popularity": 0.15,
                    "business_coefficient": 0.5,
                    "nightlife_coefficient": 0.3,
                    "residential_coefficient": 0.4,
                    "address_templates": [
                        "{} Fashion Avenue",
                        "{} Commerce Street",
                        "{} Mall Boulevard",
                        "{} Retail Row",
                        "{} Market Street"
                    ]
                },
                "nightlife_area": {
                    "name": "Nightlife District",
                    "center": (40.7200, -74.0050),
                    "radius": 1.5,
                    "popularity": 0.1,
                    "business_coefficient": 0.2,
                    "nightlife_coefficient": 0.9,
                    "residential_coefficient": 0.3,
                    "address_templates": [
                        "{} Club Street",
                        "{} Party Avenue",
                        "{} Entertainment Boulevard",
                        "{} Bar Street",
                        "{} Night Avenue"
                    ]
                },
                "university": {
                    "name": "University Area",
                    "center": (40.7400, -74.0250),
                    "radius": 1.5,
                    "popularity": 0.1,
                    "business_coefficient": 0.4,
                    "nightlife_coefficient": 0.4,
                    "residential_coefficient": 0.5,
                    "address_templates": [
                        "{} College Avenue",
                        "{} University Boulevard",
                        "{} Campus Drive",
                        "{} Academic Street",
                        "{} Student Center Road"
                    ]
                }
            }
    
    def get_zone_weights(self, current_time, location_type="origin"):
        """
        Calculate zone weights based on time of day and location type.
        
        Args:
            current_time: Datetime object representing the current time
            location_type: "origin" or "destination"
            
        Returns:
            dict: Zone names and their calculated weights
        """
        hour = current_time.hour
        is_business_hours = 8 <= hour <= 18
        is_evening = 18 <= hour <= 23
        is_night = 0 <= hour < 6
        is_morning = 6 <= hour < 8
        is_weekend = current_time.weekday() >= 5
        
        weights = {}
        
        for zone_name, zone in self.zones.items():
            base_weight = zone["popularity"]
            
            # Apply time-based adjustments
            if is_business_hours and not is_weekend:
                if location_type == "origin":
                    # During business hours, residential areas are common origins
                    weight = base_weight * (
                        0.5 + 0.5 * zone["residential_coefficient"] + 
                        0.3 * zone["business_coefficient"]
                    )
                else:  # destination
                    # During business hours, business areas are common destinations
                    weight = base_weight * (
                        0.3 + 0.7 * zone["business_coefficient"]
                    )
            elif is_morning and not is_weekend:
                if location_type == "origin":
                    # Morning: residential areas are strong origins
                    weight = base_weight * (
                        0.3 + 0.7 * zone["residential_coefficient"]
                    )
                else:  # destination
                    # Morning: business areas are strong destinations
                    weight = base_weight * (
                        0.3 + 0.7 * zone["business_coefficient"]
                    )
            elif is_evening:
                if location_type == "origin":
                    # Evening: business areas are common origins
                    weight = base_weight * (
                        0.4 + 0.6 * zone["business_coefficient"] if not is_weekend else
                        0.5 + 0.5 * zone["nightlife_coefficient"]
                    )
                else:  # destination
                    # Evening: residential and nightlife areas are common destinations
                    weight = base_weight * (
                        0.3 + 0.4 * zone["residential_coefficient"] + 
                        0.3 * zone["nightlife_coefficient"] if not is_weekend else
                        0.2 + 0.2 * zone["residential_coefficient"] + 
                        0.6 * zone["nightlife_coefficient"]
                    )
            elif is_night:
                if location_type == "origin":
                    # Night: nightlife areas are strong origins
                    weight = base_weight * (
                        0.2 + 0.8 * zone["nightlife_coefficient"]
                    )
                else:  # destination
                    # Night: residential areas are strong destinations
                    weight = base_weight * (
                        0.2 + 0.8 * zone["residential_coefficient"]
                    )
            else:  # weekend daytime or other times
                # More balanced but still reflect zone type
                weight = base_weight * (
                    0.5 + 0.2 * zone["residential_coefficient"] + 
                    0.2 * zone["business_coefficient"] + 
                    0.1 * zone["nightlife_coefficient"]
                )
            
            # Special case for airport
            if zone_name == "airport":
                # Adjust airport weights based on time
                if 5 <= hour <= 9 or 17 <= hour <= 21:  # Peak travel times
                    weight *= 1.5
                elif 0 <= hour < 5:  # Very early morning
                    weight *= 0.3
            
            weights[zone_name] = weight
        
        return weights
    
    def select_zone(self, current_time, location_type="origin"):
        """
        Select a zone based on weights.
        
        Args:
            current_time: Datetime object
            location_type: "origin" or "destination"
            
        Returns:
            str: Selected zone name
        """
        weights = self.get_zone_weights(current_time, location_type)
        zone_names = list(weights.keys())
        zone_weights = [weights[zone] for zone in zone_names]
        
        # Normalize weights
        total_weight = sum(zone_weights)
        normalized_weights = [w/total_weight for w in zone_weights]
        
        return random.choices(zone_names, weights=normalized_weights, k=1)[0]
    
    def generate_location(self, current_time, location_type="origin", origin=None, destination=None):
        """
        Generate a realistic location based on time of day and location type.
        
        Args:
            current_time: Datetime object representing the current time
            location_type: "origin" or "destination"
            origin: If generating a destination, the origin location (optional)
            destination: If generating an origin, the destination location (optional)
            
        Returns:
            dict: Location object with latitude, longitude, address, and city
        """
        if location_type == "destination" and origin:
            # If we have an origin, select a destination that makes sense
            # Sometimes people travel within the same zone, sometimes to different zones
            origin_zone = self._find_zone(origin["latitude"], origin["longitude"])
            
            if origin_zone and random.random() < 0.3:  # 30% chance to stay in same zone
                selected_zone = origin_zone
            else:
                # For the other 70%, pick weighted zone but avoid the same one
                weights = self.get_zone_weights(current_time, "destination")
                
                if origin_zone:
                    # Slightly reduce weight of the origin zone to encourage different destinations
                    weights[origin_zone] = weights.get(origin_zone, 0) * 0.5
                
                zone_names = list(weights.keys())
                zone_weights = [weights[zone] for zone in zone_names]
                
                # Normalize weights
                total_weight = sum(zone_weights)
                normalized_weights = [w/total_weight for w in zone_weights]
                
                selected_zone = random.choices(zone_names, weights=normalized_weights, k=1)[0]
        elif location_type == "origin" and destination:
            # Similar logic if we're generating an origin based on a known destination
            dest_zone = self._find_zone(destination["latitude"], destination["longitude"])
            
            if dest_zone and random.random() < 0.3:  # 30% chance to stay in same zone
                selected_zone = dest_zone
            else:
                weights = self.get_zone_weights(current_time, "origin")
                
                if dest_zone:
                    weights[dest_zone] = weights.get(dest_zone, 0) * 0.5
                
                zone_names = list(weights.keys())
                zone_weights = [weights[zone] for zone in zone_names]
                
                total_weight = sum(zone_weights)
                normalized_weights = [w/total_weight for w in zone_weights]
                
                selected_zone = random.choices(zone_names, weights=normalized_weights, k=1)[0]
        else:
            # No constraints, just select based on time-appropriate weights
            selected_zone = self.select_zone(current_time, location_type)
        
        # Get the selected zone details
        zone = self.zones[selected_zone]
        
        # Generate a random point within the zone (using normal distribution for more clustering)
        # Standard deviation is set to create a nice distribution within the radius
        std_dev = zone["radius"] / 3 / 111  # Convert km to degrees (approx) and scale for normal distribution
        
        lat = random.normalvariate(zone["center"][0], std_dev)
        lng = random.normalvariate(zone["center"][1], std_dev)
        
        # Make sure the point is not too far from the center
        max_distance = zone["radius"] / 111  # Convert km to degrees (approximately)
        while self.haversine_distance(lat, lng, zone["center"][0], zone["center"][1]) > max_distance:
            lat = random.normalvariate(zone["center"][0], std_dev)
            lng = random.normalvariate(zone["center"][1], std_dev)
        
        # Generate a plausible address
        address = self._generate_address(zone)
        
        return {
            "latitude": lat,
            "longitude": lng,
            "address": address,
            "city": self.city_name
        }
    
    def _generate_address(self, zone):
        """Generate a plausible address for a zone"""
        # Select a template from the zone's address templates
        if "address_templates" in zone and zone["address_templates"]:
            template = random.choice(zone["address_templates"])
            # Generate a random street number
            number = random.randint(1, 9999)
            return template.format(number)
        else:
            # Fallback if no templates are defined
            return f"{random.randint(1, 9999)} {zone['name']} Area"
    
    def _find_zone(self, lat, lng):
        """Find which zone a point belongs to"""
        for zone_name, zone in self.zones.items():
            distance = self.haversine_distance(lat, lng, zone["center"][0], zone["center"][1])
            if distance <= zone["radius"] / 111:  # Convert km to degrees (approximately)
                return zone_name
        return None
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    def calculate_distance(self, location1, location2):
        """Calculate distance between two locations in kilometers"""
        return self.haversine_distance(
            location1["latitude"], location1["longitude"],
            location2["latitude"], location2["longitude"]
        )
    
    def save_to_json(self, filename):
        """Save the city map to a JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                "city_name": self.city_name,
                "zones": self.zones,
                "bounds": {
                    "lat_min": self.lat_min,
                    "lat_max": self.lat_max,
                    "lng_min": self.lng_min,
                    "lng_max": self.lng_max
                }
            }, f, indent=2)
        
        return filename

# Example usage
if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Test the geographic model')
    parser.add_argument('--city', type=str, default='San Francisco', help='City name')
    parser.add_argument('--save', type=str, help='Save city map to JSON file')
    
    args = parser.parse_args()
    
    city = CityMap(args.city)
    
    # Test generating locations at different times
    times = [
        datetime(2023, 6, 15, 8, 0),   # Weekday morning
        datetime(2023, 6, 15, 12, 0),  # Weekday noon
        datetime(2023, 6, 15, 17, 30), # Weekday evening rush
        datetime(2023, 6, 15, 22, 0),  # Weekday night
        datetime(2023, 6, 17, 11, 0),  # Weekend morning
        datetime(2023, 6, 17, 21, 0),  # Weekend night
    ]
    
    print(f"Testing Geographic Model for {args.city}")
    print("=" * 40)
    
    for t in times:
        print(f"\nTime: {t.strftime('%A %H:%M')}")
        
        # Get origin location
        origin = city.generate_location(t, "origin")
        print(f"Origin: {origin['address']}, {origin['city']} ({origin['latitude']:.4f}, {origin['longitude']:.4f})")
        
        # Get destination based on origin
        destination = city.generate_location(t, "destination", origin=origin)
        print(f"Destination: {destination['address']}, {destination['city']} ({destination['latitude']:.4f}, {destination['longitude']:.4f})")
        
        # Calculate distance
        distance = city.calculate_distance(origin, destination)
        print(f"Distance: {distance:.2f} km")
    
    if args.save:
        saved_file = city.save_to_json(args.save)
        print(f"\nCity map saved to {saved_file}")
