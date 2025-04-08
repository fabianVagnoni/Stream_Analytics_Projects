"""
Main script for Ride-Hailing Data Generation

This script ties together all components of the ride-hailing data generator,
coordinating the generation of users, drivers, rides, special events, and 
output in both JSON and AVRO formats.

Usage:
    python3 main.py --output ./data --start-date start_date --end-date end_date --drivers num_drivers --users num_users --base-demand hourly_demand --city city_name --num-concerts num_concerts --num-sports num_sports --num-weather num_weather --no-special-events --json-only --avro-only --batch-size batch_size --random-seed seed
"""

import os
import json
import time
import argparse
import random
import logging
from datetime import datetime, timedelta
from confluent_kafka import Producer
import fastavro
import io
import ssl


# Import module components
from generate_static_data import generate_users, generate_drivers
from geographic_model import CityMap
from temporal_patterns import DemandModel, TrafficModel, generate_spanish_holidays
from ride_simulator import RideSimulator
from special_events import SpecialEventsGenerator
from avro_serializer import AvroSerializer, UserDriverSerializer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Azure Event Hubs configuration
EVENT_HUB_NAMESPACE = "iesstsabbadbaa-grp-01-05"
RIDE_EVENT_HUB_NAME = "grp04-ride-events"  
SPECIAL_EVENT_HUB = "grp04-special-events"
RIDES_PRIMARY_CONNECTION_STRING = os.getenv("RIDES_PRIMARY_CONNECTION_STRING")
SPECIAL_PRIMARY_CONNECTION_STRING = os.getenv("SPECIAL_PRIMARY_CONNECTION_STRING")
BOOTSTRAP_SERVERS = f"{EVENT_HUB_NAMESPACE}.servicebus.windows.net:9093"

# Get schemas
with open("./schemas/ride_datafeed_schema.json", 'r') as f:
            schema = json.load(f)
PARSED_RIDE_SCHEMA = fastavro.parse_schema(schema)

with open("./schemas/special_events_schema.json", 'r') as f:
    special_schema = json.load(f)
PARSED_SPECIAL_SCHEMA = fastavro.parse_schema(special_schema)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate ride-hailing data')
    
    # Output options
    parser.add_argument('--output', type=str, default='./data', 
                        help='Output directory for generated data')
    parser.add_argument('--json-only', action='store_true',
                        help='Generate only JSON output (no AVRO)')
    parser.add_argument('--avro-only', action='store_true',
                        help='Generate only AVRO output (no JSON)')
    
    # Time range options
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to simulate (if not using start/end dates)')
    
    # Volume options
    parser.add_argument('--users', type=int, default=1000,
                        help='Number of users to generate')
    parser.add_argument('--drivers', type=int, default=500,
                        help='Number of drivers to generate')
    parser.add_argument('--base-demand', type=int, default=100,
                        help='Base hourly demand for rides')
    
    # Schema options
    parser.add_argument('--ride-schema', type=str, default='./schemas/ride_datafeed_schema.json',
                        help='Path to ride event schema')
    parser.add_argument('--user-driver-schema', type=str, default='./schemas/riders_drivers_avro-schemas.json',
                        help='Path to user/driver schema')
    
    # City options
    parser.add_argument('--city', type=str, default='Madrid',
                        help='City name for simulation')
    
    # Special events options
    parser.add_argument('--no-special-events', action='store_true',
                        help='Disable generation of special events')
    parser.add_argument('--num-concerts', type=int, default=1,
                        help='Number of concert events to generate')
    parser.add_argument('--num-sports', type=int, default=1,
                        help='Number of sports events to generate')
    parser.add_argument('--num-weather', type=int, default=1,
                        help='Number of weather events to generate')
    
    # Processing options
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='Batch size for processing/saving events')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--stream-to-eventhubs', action='store_true', help='Stream ride events to Azure Event Hubs in real time')
    
    return parser.parse_args()

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")
    else:
        print(f"Delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

def serialize_event(schema, record):
    '''
    try:
        # Convert the record (a dict) into a JSON string and then encode to bytes
        return json.dumps(record).encode('utf-8')
    except Exception as e:
        logger.error(f"Serialization error: {e}")
        raise
    '''
    try:
        with io.BytesIO() as buf:
            fastavro.schemaless_writer(buf, schema, record)
            return buf.getvalue()

    except Exception as e:
        logger.error(f"Serialization error: {e}")
        raise

def create_producer(connection_string):
    """Initialize Kafka producer for Azure Event Hubs with specified schema and connection string."""
    #ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    #ssl_context.check_hostname = False
    #ssl_context.verify_mode = ssl.CERT_NONE

    try:
        conf = {'bootstrap.servers': BOOTSTRAP_SERVERS,
            'security.protocol':'SASL_SSL',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': '$ConnectionString',
            'sasl.password': connection_string,  
            'client.id': 'Events-Producer',
            #ssl_context=ssl_context,
            #api_version=(0, 10, 2)}
        }

        producer = Producer(**conf)
        logger.info("Kafka producer initialized for Event Hubs")
        return producer
    except Exception as e:
        logger.error(f"Failed to initialize producer: {e}")
        raise


def generate_simulation_data(args, special_producer=None, ride_producer=None):
    """
    Generate all data for the ride-hailing simulation.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Paths to generated data files
    """
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set random seed if provided
    if args.random_seed is not None:
        random.seed(args.random_seed)
        logger.info(f"Set random seed to {args.random_seed}")
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.fromisoformat(args.start_date)
        end_date = datetime.fromisoformat(args.end_date)
    else:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=args.days)
    
    logger.info(f"Generating data for period: {start_date.date()} to {end_date.date()}")
    
    # Track output files
    output_files = {
        "json": {},
        "avro": {}
    }
    
    # 1. Generate static data (users and drivers)
    logger.info(f"Generating {args.users} users and {args.drivers} drivers for {args.city}")
    
    users_static, users_dynamic = generate_users(args.users, args.city)
    drivers_static, drivers_dynamic = generate_drivers(args.drivers, args.city)
    
    # Save static data as JSON
    if not args.avro_only:
        users_static_file = os.path.join(args.output, 'users_static.json')
        users_dynamic_file = os.path.join(args.output, 'users_dynamic.json')
        drivers_static_file = os.path.join(args.output, 'drivers_static.json')
        drivers_dynamic_file = os.path.join(args.output, 'drivers_dynamic.json')
        
        with open(users_static_file, 'w') as f:
            json.dump(users_static, f, indent=2)
        
        with open(users_dynamic_file, 'w') as f:
            json.dump(users_dynamic, f, indent=2)
        
        with open(drivers_static_file, 'w') as f:
            json.dump(drivers_static, f, indent=2)
        
        with open(drivers_dynamic_file, 'w') as f:
            json.dump(drivers_dynamic, f, indent=2)
        
        output_files["json"]["users_static"] = users_static_file
        output_files["json"]["users_dynamic"] = users_dynamic_file
        output_files["json"]["drivers_static"] = drivers_static_file
        output_files["json"]["drivers_dynamic"] = drivers_dynamic_file
        
        logger.info(f"Saved user/driver JSON data to {args.output}")
    
    # Save static data as AVRO
    if not args.json_only:
        # Initialize the UserDriverSerializer
        ud_serializer = UserDriverSerializer(args.user_driver_schema)
        
        users_static_avro = os.path.join(args.output, 'users_static.avro')
        users_dynamic_avro = os.path.join(args.output, 'users_dynamic.avro')
        drivers_static_avro = os.path.join(args.output, 'drivers_static.avro')
        drivers_dynamic_avro = os.path.join(args.output, 'drivers_dynamic.avro')
        
        ud_serializer.serialize_data(users_static, 'UserStatic', users_static_avro)
        ud_serializer.serialize_data(users_dynamic, 'UserDynamic', users_dynamic_avro)
        ud_serializer.serialize_data(drivers_static, 'DriverStatic', drivers_static_avro)
        ud_serializer.serialize_data(drivers_dynamic, 'DriverDynamic', drivers_dynamic_avro)
        
        output_files["avro"]["users_static"] = users_static_avro
        output_files["avro"]["users_dynamic"] = users_dynamic_avro
        output_files["avro"]["drivers_static"] = drivers_static_avro
        output_files["avro"]["drivers_dynamic"] = drivers_dynamic_avro
        
        logger.info(f"Saved user/driver AVRO data to {args.output}")
    

    # 2. Initialize models
    # Create city map
    city_map = CityMap(args.city)
    city_map_file = os.path.join(args.output, f'{args.city.lower().replace(" ", "_")}_map.json')
    city_map.save_to_json(city_map_file)
    output_files["json"]["city_map"] = city_map_file
    
    # Create demand model with holidays
    current_year = start_date.year
    holidays = generate_spanish_holidays(current_year)
    # Add holidays for next year if simulation crosses year boundary
    if end_date.year > current_year:
        holidays.extend(generate_spanish_holidays(current_year + 1))
    
    demand_model = DemandModel(base_demand=args.base_demand, holidays=holidays)
    traffic_model = TrafficModel(city_name=args.city)
    
    # 3. Create ride simulator
    ride_simulator = RideSimulator(
        users_dynamic, 
        drivers_dynamic, 
        city_map, 
        demand_model, 
        traffic_model
    )
    
    # 4. Add special events if enabled
    if not args.no_special_events:
        logger.info("Generating special events")
        
        special_events_generator = SpecialEventsGenerator(
            city_map,
            demand_model,
            traffic_model,
            ride_simulator
        )
        
        # Add concert events
        for i in range(args.num_concerts):
            # Spread concerts throughout the simulation period
            days_range = (end_date - start_date).days
            event_day = random.randint(0, days_range - 1)
            event_date = start_date + timedelta(days=event_day)
            
            # Set to evening hours (6-8 PM)
            event_hour = random.randint(18, 20)
            event_date = event_date.replace(hour=event_hour, minute=0)
            
            # Select a random zone, preferring downtown or entertainment areas
            zone_weights = {
                zone: 3.0 if "downtown" in zone.lower() or "nightlife" in zone.lower() else 1.0
                for zone in city_map.zones.keys()
            }
            zones = list(zone_weights.keys())
            zone_weight_values = [zone_weights[z] for z in zones]
            
            venue_zone = random.choices(zones, weights=zone_weight_values, k=1)[0]
            
            # Generate attendance between 3000-15000
            attendees = random.randint(3000, 15000)
            
            concert_event = special_events_generator.create_concert_event(
                event_date,
                venue_zone=venue_zone,
                attendees=attendees,
                name=f"Concert Event {i+1}"
            )
            
            if args.stream_to_eventhubs and special_producer:
                avro_bytes = serialize_event(PARSED_SPECIAL_SCHEMA, concert_event)
                special_producer.produce(topic=SPECIAL_EVENT_HUB, value=avro_bytes, callback=delivery_report)
                special_producer.poll(0)
                special_producer.flush()
            logger.info(f"Added concert event on {event_date} at {city_map.zones[venue_zone]['name']}")
            
        
        # Add sports events
        for i in range(args.num_sports):
            # Spread sports events throughout the simulation period
            days_range = (end_date - start_date).days
            event_day = random.randint(0, days_range - 1)
            event_date = start_date + timedelta(days=event_day)
            
            # Set to afternoon/evening hours (12-7 PM)
            event_hour = random.randint(12, 19)
            event_date = event_date.replace(hour=event_hour, minute=0)
            
            # Select a random zone
            venue_zone = random.choice(list(city_map.zones.keys()))
            
            # Generate attendance between 10000-40000
            attendees = random.randint(10000, 40000)
            
            sports_event = special_events_generator.create_sports_event(
                event_date,
                venue_zone=venue_zone,
                attendees=attendees
            )
            if args.stream_to_eventhubs and special_producer:
                avro_bytes = serialize_event(PARSED_SPECIAL_SCHEMA, sports_event)
                special_producer.produce(topic=SPECIAL_EVENT_HUB, value=avro_bytes, callback=delivery_report)
                special_producer.poll(0)
                special_producer.flush()
            
            logger.info(f"Added sports event on {event_date} at {city_map.zones[venue_zone]['name']}")
        
        # Add weather events
        for i in range(args.num_weather):
            # Spread weather events throughout the simulation period
            days_range = (end_date - start_date).days
            event_day = random.randint(0, days_range - 1)
            event_date = start_date + timedelta(days=event_day)
            
            # Weather typically starts in early morning
            event_hour = random.randint(5, 9)
            event_date = event_date.replace(hour=event_hour, minute=0)
            
            # Duration between 4 and 24 hours
            duration_hours = random.randint(4, 24)
            
            # Severity distribution: 20% light, 60% medium, 20% severe
            severity = random.choices(
                ["light", "medium", "severe"],
                weights=[0.2, 0.6, 0.2],
                k=1
            )[0]
            
            special_events_generator.create_weather_event(
                event_date,
                duration_hours=duration_hours,
                severity=severity
            )
            
            logger.info(f"Added {severity} weather event on {event_date} for {duration_hours} hours")
        
        # Add system outage
        outage_day = random.randint(0, (end_date - start_date).days - 1)
        outage_date = start_date + timedelta(days=outage_day)
        
        # Outages typically during business hours
        outage_hour = random.randint(9, 17)
        outage_date = outage_date.replace(hour=outage_hour, minute=0)
        
        # Duration between 30 minutes and 3 hours
        duration_minutes = random.randint(30, 180)
        
        special_events_generator.create_system_outage(
            outage_date,
            duration_minutes=duration_minutes,
            severity=random.choice(["partial", "complete"])
        )
        
        logger.info(f"Added system outage on {outage_date} for {duration_minutes} minutes")
        
        # Add fraud patterns
        special_events_generator.create_fraud_patterns(
            start_date,
            end_date,
            num_fraud_users=min(5, max(1, int(args.users * 0.01))),  # At least 1, max 5
            num_fraud_drivers=min(3, max(1, int(args.drivers * 0.01)))  # At least 1, max 3
        )
        
        logger.info(f"Added fraudulent patterns")
        
        # Save special events data
        special_events_file = os.path.join(args.output, 'special_events.json')
        special_events_generator.save_to_json(special_events_file)
        output_files["json"]["special_events"] = special_events_file
    
    # 5. Generate ride events
    logger.info(f"Generating rides from {start_date} to {end_date}")
    rides_json_file = os.path.join(args.output, 'ride_events.json')
    rides_avro_file = os.path.join(args.output, 'ride_events.avro')

    if args.stream_to_eventhubs and ride_producer:
        event_count = 0
        for event in ride_simulator.generate_rides(start_date, end_date): #[:200]
            time.sleep(0.01) # Increase this if Azure complains
            avro_bytes = serialize_event(PARSED_RIDE_SCHEMA, event)
            ride_producer.produce(topic=RIDE_EVENT_HUB_NAME, value=avro_bytes)
            event_count += 1
            if event_count % args.batch_size == 0:
                time.sleep(2)
                ride_producer.flush()
                logger.info(f"Streamed {event_count} ride events to Event Hubs")
        ride_producer.flush()
        logger.info(f"Streamed total of {event_count} ride events to Event Hubs")
    else:
        event_count = ride_simulator.generate_rides(start_date, end_date, output_file=rides_json_file, batch_size=args.batch_size)
        output_files["json"]["ride_events"] = rides_json_file
        logger.info(f"Generated {event_count} ride events, saved to {rides_json_file}")

    if not args.json_only and not args.stream_to_eventhubs:
        ride_serializer = AvroSerializer(args.ride_schema)
        with open(rides_json_file, 'r') as f:
            events = json.load(f)
        ride_serializer.serialize_to_file(events, rides_avro_file)
        output_files["avro"]["ride_events"] = rides_avro_file
        logger.info(f"Serialized ride events to AVRO: {rides_avro_file}")
    
    # 6. Convert to AVRO if needed
    if not args.json_only:
        ride_serializer = AvroSerializer(args.ride_schema)
        rides_avro_file = os.path.join(args.output, 'ride_events.avro')
        
        if args.avro_only:
            # Serialize events in memory
            ride_serializer.serialize_to_file(events, rides_avro_file)
        else:
            # Load from JSON file and serialize
            with open(rides_json_file, 'r') as f:
                events = json.load(f)
            
            ride_serializer.serialize_to_file(events, rides_avro_file)
        
        output_files["avro"]["ride_events"] = rides_avro_file
        logger.info(f"Serialized ride events to AVRO format: {rides_avro_file}")
    
    # 7. Generate summary statistics
    total_events = sum(1 for line in open(rides_json_file)) - 2  # Subtract array brackets
    
    summary = {
        "simulation_period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": (end_date - start_date).days
        },
        "users": {
            "total": len(users_static),
            "dynamic_stats": {
                "avg_rides_taken": sum(u["rides_taken"] for u in users_dynamic) / len(users_dynamic),
                "avg_money_spent": sum(u["money_spent"] for u in users_dynamic) / len(users_dynamic),
                "avg_rating_given": sum(u["avg_rating_given"] for u in users_dynamic) / len(users_dynamic),
                "avg_cancellation_rate": sum(u["cancellation_rate"] for u in users_dynamic) / len(users_dynamic)
            }
        },
        "drivers": {
            "total": len(drivers_static),
            "dynamic_stats": {
                "avg_rides": sum(d["no_of_rides"] for d in drivers_dynamic) / len(drivers_dynamic),
                "avg_rating": sum(d["rating"] for d in drivers_dynamic) / len(drivers_dynamic),
                "avg_money_earned": sum(d["money_earned"] for d in drivers_dynamic) / len(drivers_dynamic),
                "avg_cancellation_rate": sum(d["cancellation_rate"] for d in drivers_dynamic) / len(drivers_dynamic)
            }
        },
        "events": {
            "total": total_events,
            "per_day": total_events / (end_date - start_date).days,
            "special_events": 0 if args.no_special_events else len(special_events_generator.special_events)
        },
        "output_files": output_files
    }
    
    # Save summary
    summary_file = os.path.join(args.output, 'simulation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved simulation summary to {summary_file}")
    
    return output_files

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info("Starting ride-hailing data generation")
    logger.info(f"Output directory: {args.output}")

    ride_producer = None
    special_producer = None
    if args.stream_to_eventhubs:
        if not RIDES_PRIMARY_CONNECTION_STRING or not SPECIAL_PRIMARY_CONNECTION_STRING:
            logger.error("Missing connection strings. Set RIDE_PRIMARY_CONNECTION_STRING and SPECIAL_PRIMARY_CONNECTION_STRING environment variables.")
            return 1
        ride_producer = create_producer(RIDES_PRIMARY_CONNECTION_STRING)
        special_producer = create_producer(SPECIAL_PRIMARY_CONNECTION_STRING)
    
    try:
        output_files = generate_simulation_data(args, special_producer=special_producer, ride_producer=ride_producer)
        
        logger.info("Data generation completed successfully")
        logger.info(f"Generated files: {list(output_files['json'].keys()) + list(output_files['avro'].keys())}")
        
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
