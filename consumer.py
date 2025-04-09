import os
import subprocess
import fastavro
import confluent_kafka
import time
import requests
import re
from packaging import version

# Check if running on Windows or Linux
is_windows = os.name == 'nt'

# LINUX VERSION
if not is_windows:
    # Get the latest Spark version
    spark_version = subprocess.run(
        "curl -s https://downloads.apache.org/spark/ | grep -o 'spark-3\\.[0-9]\\+\\.[0-9]\\+' | sort -V | tail -1",
        shell=True, capture_output=True, text=True
    ).stdout.strip()
    print(f"Spark version: {spark_version}")
    
    spark_release = spark_version
    hadoop_version = 'hadoop3'
    start = time.time()
    
    os.environ['SPARK_RELEASE'] = spark_release
    os.environ['HADOOP_VERSION'] = hadoop_version
    
    # Define paths
    spark_file = f"{spark_release}-bin-{hadoop_version}.tgz"
    spark_dir = f"{spark_release}-bin-{hadoop_version}"
    current_dir = os.getcwd()
    spark_home_dir = os.path.join(current_dir, spark_dir)
    
    # Check if Spark is already downloaded and extracted
    if os.path.exists(spark_home_dir) and os.path.isdir(spark_home_dir):
        print(f"Spark already downloaded and extracted at: {spark_home_dir}")
    else:
        # Install Java if needed
        try:
            java_check = subprocess.run("java -version", shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if java_check.returncode != 0:
                print("Installing Java...")
                subprocess.run("apt-get install openjdk-8-jdk-headless -qq > /dev/null", shell=True, check=True)
            else:
                print("Java is already installed")
        except Exception as e:
            print(f"Error checking/installing Java: {e}")
            print("Attempting to install Java...")
            subprocess.run("apt-get install openjdk-8-jdk-headless -qq > /dev/null", shell=True, check=True)
        
        # Download Spark if not already downloaded
        if not os.path.exists(spark_file):
            print(f"Downloading Spark from https://archive.apache.org/dist/spark/{spark_release}/{spark_file}")
            subprocess.run(f"wget -q https://archive.apache.org/dist/spark/{spark_release}/{spark_file}", shell=True, check=True)
            print(f"Downloaded {spark_file}")
        else:
            print(f"Spark archive {spark_file} already exists, skipping download")
        
        # Extract the Spark archive
        subprocess.run(f"tar xf {spark_file}", shell=True, check=True)
        print(f"Extracted {spark_file}")
    
    # Set environment variables
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["SPARK_HOME"] = spark_home_dir
    print(f"Set SPARK_HOME to {spark_home_dir}")

# WINDOWS VERSION 
if is_windows:
    # Use Python requests to get the latest Spark version
    try:
        response = requests.get("https://downloads.apache.org/spark/")
        matches = re.findall(r'spark-3\.\d+\.\d+', response.text)
        if matches:
            spark_version = max(matches, key=lambda x: version.parse(x.replace('spark-', '')))
        else:
            spark_version = "spark-3.5.1"  # Fallback version
    except Exception as e:
        print(f"Error fetching Spark version: {e}")
        spark_version = "spark-3.5.1"  # Fallback version
    
    print(f"Spark version: {spark_version}")
    
    # Set variables
    spark_release = spark_version
    hadoop_version = 'hadoop3'
    start = time.time()
    
    # Set environment variables
    os.environ['SPARK_RELEASE'] = spark_release
    os.environ['HADOOP_VERSION'] = hadoop_version
    
    # Define paths
    spark_file = f"{spark_release}-bin-{hadoop_version}.tgz"
    spark_dir = f"{spark_release}-bin-{hadoop_version}"
    current_dir = os.getcwd()
    spark_home_dir = os.path.join(current_dir, spark_dir)
    
    # Set up Hadoop for Windows
    hadoop_dir = os.path.join(current_dir, "hadoop")
    hadoop_bin_dir = os.path.join(hadoop_dir, "bin")
    winutils_path = os.path.join(hadoop_bin_dir, "winutils.exe")
    hadoop_dll_path = os.path.join(hadoop_bin_dir, "hadoop.dll")
    
    # Create Hadoop directory if it doesn't exist
    if not os.path.exists(hadoop_dir):
        os.makedirs(hadoop_dir, exist_ok=True)
    if not os.path.exists(hadoop_bin_dir):
        os.makedirs(hadoop_bin_dir, exist_ok=True)
    
    # Set HADOOP_HOME environment variable
    os.environ["HADOOP_HOME"] = hadoop_dir
    print(f"Set HADOOP_HOME to {hadoop_dir}")
    
    # Add hadoop/bin to PATH
    os.environ["PATH"] = os.environ["PATH"] + os.pathsep + hadoop_bin_dir
    
    # Download winutils.exe and hadoop.dll if they don't exist
    files_to_download = [
        {
            "name": "winutils.exe",
            "path": winutils_path,
            "url": "https://github.com/cdarlint/winutils/raw/master/hadoop-3.2.2/bin/winutils.exe"
        },
        {
            "name": "hadoop.dll",
            "path": hadoop_dll_path,
            "url": "https://github.com/cdarlint/winutils/raw/master/hadoop-3.2.2/bin/hadoop.dll"
        }
    ]
    
    for file_info in files_to_download:
        if not os.path.exists(file_info["path"]):
            try:
                print(f"Downloading {file_info['name']}...")
                download_cmd = f'powershell -Command "Invoke-WebRequest -Uri \'{file_info["url"]}\' -OutFile \'{file_info["path"]}\'"'
                subprocess.run(download_cmd, shell=True, check=True)
                print(f"Downloaded {file_info['name']} to {file_info['path']}")
            except Exception as e:
                print(f"Error downloading {file_info['name']}: {e}")
                print(f"Please download {file_info['name']} manually and place it in hadoop/bin directory")
        else:
            print(f"{file_info['name']} already exists at {file_info['path']}")
    
    # Check if Spark is already downloaded and extracted
    if os.path.exists(spark_home_dir) and os.path.isdir(spark_home_dir):
        print(f"Spark already downloaded and extracted at: {spark_home_dir}")
    else:
        # Download and install Spark using PowerShell
        spark_url = f"https://archive.apache.org/dist/spark/{spark_release}/{spark_release}-bin-{hadoop_version}.tgz"
        print(f"Downloading Spark from {spark_url}")
        
        # Use PowerShell to download the file
        download_cmd = f'powershell -Command "Invoke-WebRequest -Uri \'{spark_url}\' -OutFile \'{spark_file}\'"'
        
        try:
            # Download Spark if not already downloaded
            if not os.path.exists(spark_file):
                subprocess.run(download_cmd, shell=True, check=True)
                print(f"Downloaded {spark_file}")
            else:
                print(f"Spark archive {spark_file} already exists, skipping download")
            
            # Extract using 7-Zip if available, otherwise try tar if available
            if os.path.exists("C:\\Program Files\\7-Zip\\7z.exe"):
                extract_cmd = f'"C:\\Program Files\\7-Zip\\7z.exe" x {spark_file}'
            else:
                # Use Windows tar if available (Windows 10 1803 and later)
                extract_cmd = f'tar -xf {spark_file}'
            
            subprocess.run(extract_cmd, shell=True, check=True)
            print(f"Extracted {spark_file}")
            
        except Exception as e:
            print(f"Error setting up Spark: {e}")
            print("Please install Spark manually and set SPARK_HOME environment variable")
    
    # Set SPARK_HOME to the extracted directory
    os.environ["SPARK_HOME"] = spark_home_dir
    print(f"Set SPARK_HOME to {spark_home_dir}")

try:
    import findspark
    import pyspark
    
    findspark.init()
    
    from pyspark.sql import SparkSession
    from pyspark.sql.avro.functions import from_avro
    from pyspark.sql.functions import col
    
    print("Successfully imported PySpark")
except Exception as e:
    print(f"Error importing PySpark: {e}")

# Define Azure credentials
event_hub_namespace = 'iesstsabbadbaa-grp-01-05'

rides_eventhub_name = 'grp04-ride-events'
rides_consumer_eventhub_connection_str = 'Endpoint=sb://iesstsabbadbaa-grp-01-05.servicebus.windows.net/;SharedAccessKeyName=Consumer;SharedAccessKey=iNowxPjC+fG9CrLnklmDTAy/J1n0e9Wpe+AEhC107ys=;EntityPath=grp04-ride-events'

specials_eventhub_name = 'grp04-special-events'
specials_consumer_eventhub_connection_str = 'Endpoint=sb://iesstsabbadbaa-grp-01-05.servicebus.windows.net/;SharedAccessKeyName=Consumer;SharedAccessKey=tJYUHSWabtnBVNOhc5TgJMHz1vtPw1NqC+AEhH6h8V4=;EntityPath=grp04-special-events'

print(f"\nEvent Hub Namespace: {event_hub_namespace}")
print(f"Rides Event Hub Name: {rides_eventhub_name}")
print(f"Rides Consumer Event Hub Connection String: {rides_consumer_eventhub_connection_str}")
print(f"Specials Event Hub Name: {specials_eventhub_name}")
print(f"Specials Consumer Event Hub Connection String: {specials_consumer_eventhub_connection_str}\n")

# Define the schema (from github)
print("Loading schema files...")
try:
    with open("schemas/ride_datafeed_schema.json") as f:
        schema = f.read()
        print("Successfully loaded ride schema")
except Exception as e:
    print(f"Error loading ride schema: {e}")
    raise

try:
    with open("schemas/special_events_schema.json") as e:
        special_schema = e.read()
        print("Successfully loaded special events schema")
except Exception as e:
    print(f"Error loading special events schema: {e}")
    raise

# Create a Spark session
print("Creating Spark session...")
spark = SparkSession \
    .builder \
    .appName("StreamingAVROFromKafka") \
    .config("spark.streaming.stopGracefullyOnShutdown", True) \
    .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.apache.spark:spark-avro_2.12:3.5.0') \
    .config("spark.sql.shuffle.partitions", 4) \
    .config("spark.driver.host", "localhost") \
    .config("spark.driver.bindAddress", "localhost") \
    .config("spark.executor.instances", 1) \
    .config("spark.executor.memory", "1g") \
    .config("spark.driver.memory", "1g") \
    .master("local[*]") \
    .getOrCreate()
print("Spark session created successfully")

print("Setting up Kafka configurations...")
kafkaConf_rides = {
    "kafka.bootstrap.servers": f"{event_hub_namespace}.servicebus.windows.net:9093",
    "kafka.sasl.mechanism": "PLAIN",
    "kafka.security.protocol": "SASL_SSL",
    "kafka.sasl.jaas.config": f'org.apache.kafka.common.security.plain.PlainLoginModule required username="$ConnectionString" password="{rides_consumer_eventhub_connection_str}";',
    "subscribe": rides_eventhub_name,
    "startingOffsets": "earliest",
    "kafka.request.timeout.ms": "60000",
    "kafka.session.timeout.ms": "60000",
    "failOnDataLoss": "false",
    "enable.auto.commit": "true",
    "groupIdPrefix": "Stream_Analytics_",
    "auto.commit.interval.ms": "5000"
}

kafkaConf_specials = {
    "kafka.bootstrap.servers": f"{event_hub_namespace}.servicebus.windows.net:9093",
    "kafka.sasl.mechanism": "PLAIN",
    "kafka.security.protocol": "SASL_SSL",
    "kafka.sasl.jaas.config": f'org.apache.kafka.common.security.plain.PlainLoginModule required username="$ConnectionString" password="{specials_consumer_eventhub_connection_str}";',
    "subscribe": specials_eventhub_name,
    "startingOffsets": "earliest",
    "kafka.request.timeout.ms": "60000",
    "kafka.session.timeout.ms": "60000",
    "failOnDataLoss": "false",
    "enable.auto.commit": "true",
    "groupIdPrefix": "Stream_Analytics_",
    "auto.commit.interval.ms": "5000"
}
print("Kafka configurations set up successfully")

# Read from Event Hub using Kafka
print("Starting to read from Event Hubs...")
try:
    print("Reading from Rides Event Hub...")
    df_rides = spark \
        .readStream \
        .format("kafka") \
        .options(**kafkaConf_rides) \
        .load()
    print("Successfully connected to Rides Event Hub")
    
    # Deserialize the AVRO messages from the value column
    df_rides = df_rides.select(from_avro(df_rides.value, schema).alias("ride_events"))
    print("Successfully deserialized Rides data")
    
    print("Reading from Special Events Event Hub...")
    df_specials = spark \
        .readStream \
        .format("kafka") \
        .options(**kafkaConf_specials) \
        .load()
    print("Successfully connected to Special Events Event Hub")
    
    # Deserialize the AVRO messages from the value column
    df_specials = df_specials.select(from_avro(df_specials.value, special_schema).alias("special_event"))
    print("Successfully deserialized Special Events data")
except Exception as e:
    print(f"Error connecting to Event Hubs: {e}")
    raise

# Flatten the schemas
df_rides = df_rides.select(
    col("ride_events.event_id"),
    col("ride_events.ride_id"),
    col("ride_events.event_type"),
    col("ride_events.timestamp"),
    col("ride_events.user_id"),
    col("ride_events.driver_id"),

    col("ride_events.pickup_location.latitude").alias("pickup_latitude"),
    col("ride_events.pickup_location.longitude").alias("pickup_longitude"),
    col("ride_events.pickup_location.address").alias("pickup_address"),
    col("ride_events.pickup_location.city").alias("pickup_city"),

    col("ride_events.dropoff_location.latitude").alias("dropoff_latitude"),
    col("ride_events.dropoff_location.longitude").alias("dropoff_longitude"),
    col("ride_events.dropoff_location.address").alias("dropoff_address"),
    col("ride_events.dropoff_location.city").alias("dropoff_city"),

    col("ride_events.ride_details.distance_km"),
    col("ride_events.ride_details.estimated_duration_minutes"),
    col("ride_events.ride_details.actual_duration_minutes"),
    col("ride_events.ride_details.vehicle_type"),
    col("ride_events.ride_details.base_fare"),
    col("ride_events.ride_details.surge_multiplier"),
    col("ride_events.ride_details.total_fare"),

    col("ride_events.payment_info.payment_method"),
    col("ride_events.payment_info.payment_status"),
    col("ride_events.payment_info.payment_id"),

    col("ride_events.ratings.user_to_driver_rating"),
    col("ride_events.ratings.driver_to_user_rating"),
    col("ride_events.ratings.user_comment"),
    col("ride_events.ratings.driver_comment"),

    col("ride_events.cancellation_info.canceled_by"),
    col("ride_events.cancellation_info.cancellation_reason"),
    col("ride_events.cancellation_info.cancellation_fee"),

    col("ride_events.traffic_conditions.traffic_level"),
    col("ride_events.traffic_conditions.estimated_delay_minutes"),

    col("ride_events.driver_location.latitude").alias("driver_latitude"),
    col("ride_events.driver_location.longitude").alias("driver_longitude"),
    col("ride_events.driver_location.heading").alias("driver_heading"),
    col("ride_events.driver_location.speed_kmh").alias("driver_speed_kmh"),

    col("ride_events.app_version"),
    col("ride_events.platform"),
    col("ride_events.session_id")
)

df_specials = df_specials.select(
    col("special_event.type").alias("event_type"),
    col("special_event.name").alias("event_name"),
    col("special_event.venue_zone").alias("venue_zone"),

    col("special_event.venue_location.latitude").alias("venue_latitude"),
    col("special_event.venue_location.longitude").alias("venue_longitude"),
    col("special_event.venue_location.address").alias("venue_address"),
    col("special_event.venue_location.city").alias("venue_city"),

    col("special_event.event_start").alias("event_start"),
    col("special_event.event_end").alias("event_end"),
    col("special_event.arrivals_start").alias("arrivals_start"),
    col("special_event.arrivals_end").alias("arrivals_end"),
    col("special_event.departures_start").alias("departures_start"),
    col("special_event.departures_end").alias("departures_end"),

    col("special_event.arrival_rides").alias("arrival_rides"),
    col("special_event.departure_rides").alias("departure_rides"),
    col("special_event.estimated_attendees").alias("estimated_attendees")
)

# Create checkpoint directory
os.makedirs("checkpoint", exist_ok=True)
spark.conf.set("spark.sql.streaming.checkpointLocation", "checkpoint")

print("Starting memory stream query...")
# If offset:Latest, send new events after running this cell.
query_name='all_rides'
query=df_rides.writeStream \
    .outputMode("update") \
    .format("memory") \
    .queryName(query_name) \
    .start()

print(f"Memory stream query '{query_name}' started successfully")

# Wait a few seconds for data to arrive
print("Waiting for data to arrive (10 seconds)...")
time.sleep(10)

# Status either "Processing new data" or "Getting offsets from..."
print("\nQuery status:", query.status)

# Check if any data is available
print("\nChecking for available data...")
try:
    count_df = spark.sql(f'SELECT count(*) as record_count FROM {query_name}')
    count = count_df.collect()[0]['record_count']
    print(f"Number of records received: {count}")
    
    if count > 0:
        print("\nShowing sample data:")
        spark.sql(f'SELECT * FROM {query_name}').show(5, truncate=True)
    else:
        print("No data received yet. This could be because:")
        print("1. No events are being published to the Event Hub")
        print("2. There might be authentication issues with the Event Hub connection")
        print("3. The consumer group might not have access to the events")
        print("4. The Event Hub name or connection strings might be incorrect")
except Exception as e:
    print(f"Error querying data: {e}")

print("\nStarting Parquet stream query...")
os.makedirs("output", exist_ok=True)
query_name='parquet'
query_parquet = df_rides.writeStream \
        .format("parquet") \
        .option("checkpointLocation","checkpoint2") \
        .option("path", "output") \
        .queryName(query_name) \
        .trigger(processingTime='20 seconds') \
        .start()

print(f"Parquet stream query '{query_name}' started successfully")

# Wait a few seconds for parquet files to build up
print("Waiting for Parquet files to be created (20 seconds)...")
time.sleep(20)

# List output files
output_files = os.listdir("output")
print(f"\nFiles in output directory: {len(output_files)}")
if output_files:
    print("Sample files:", output_files[:5])
else:
    print("No output files created yet")

print("\n" + "="*50)
print("STREAMING QUERIES ARE NOW RUNNING")
print("="*50)
print("Active queries:")
for q in spark.streams.active:
    print(f"  - {q.name} (ID: {q.id})")
print("\nThe script will continue running, leave it open to collect data.")
print("To stop the queries, uncomment the 'stop_queries = True' line below and run this cell again.")

# Uncomment the line below to stop all queries
# stop_queries = True
stop_queries = False

if stop_queries:
    print("Stopping all streaming queries...")
    for q in spark.streams.active:
        print(f"Stopping {q.name}...")
        q.stop()
    
    print("Stopping Spark session...")
    spark.stop()
    print("All queries and Spark session stopped.")

