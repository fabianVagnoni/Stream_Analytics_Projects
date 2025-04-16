import os
import subprocess
import fastavro
import confluent_kafka
import time
import requests
import re
from packaging import version
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, lit
from pyspark.sql.functions import window, avg, max, count, variance, expr, when, lit, collect_list
from pyspark.sql.window import Window
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, dayofyear, hour
from pyspark.sql.types import StringType, IntegerType, DoubleType, FloatType, BooleanType
from pyspark.sql.functions import count, sum, avg, max, variance, expr, when, col


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
            # spark_version = max(matches, key=lambda x: version.parse(x.replace('spark-', '')))
            sorted_versions = sorted(matches, key=lambda x: version.parse(x.replace('spark-', '')))
            spark_version = sorted_versions[-1] if sorted_versions else "spark-3.5.1"
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
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.task.cpus", "1") \
    .config("spark.kafka.consumer.cache.timeout", "60s") \
    .config("spark.kafka.consumer.max.poll.records", "500") \
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


# Prepare rides data with special events info for user profile creation
print("Preparing data for user profile vectors...")

# Create checkpoint directory
os.makedirs("checkpoint", exist_ok=True)
os.makedirs("checkpoint/rides", exist_ok=True)
os.makedirs("checkpoint/parquet", exist_ok=True)
os.makedirs("checkpoint/parquet/rides", exist_ok=True)
os.makedirs("checkpoint/parquet/specials", exist_ok=True)
os.makedirs("checkpoint/parquet/user_vectors", exist_ok=True)
spark.conf.set("spark.sql.streaming.checkpointLocation", "checkpoint")

print("Starting memory stream query...")
# If offset:Latest, send new events after running this cell.
query_name='all_rides'
query=df_rides.writeStream \
    .outputMode("update") \
    .format("memory") \
    .queryName(query_name) \
    .option("checkpointLocation", "checkpoint/rides") \
    .option("failOnDataLoss", "false") \
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
    count_rides = count_df.collect()[0]['record_count']
    columns = spark.sql(f'SELECT * FROM {query_name}').columns
    print(f"Columns: {columns}")
    print(f"Number of records received: {count_rides}")
    
    if count_rides > 0:
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
os.makedirs("output/rides", exist_ok=True)
os.makedirs("output/specials", exist_ok=True)
os.makedirs("output/user_vectors", exist_ok=True)

# Save rides data to parquet
rides_query_name = 'rides_parquet'
rides_query_parquet = df_rides.writeStream \
        .format("parquet") \
        .option("checkpointLocation","checkpoint/parquet/rides") \
        .option("path", "output/rides") \
        .option("failOnDataLoss", "false") \
        .queryName(rides_query_name) \
        .outputMode("append") \
        .trigger(processingTime='20 seconds') \
        .start()

print(f"Rides Parquet stream query '{rides_query_name}' started successfully")

# Save special events data to parquet
specials_query_name = 'specials_parquet'
specials_query_parquet = df_specials.writeStream \
        .format("parquet") \
        .option("checkpointLocation","checkpoint/parquet/specials") \
        .option("path", "output/specials") \
        .option("failOnDataLoss", "false") \
        .queryName(specials_query_name) \
        .outputMode("append") \
        .trigger(processingTime='20 seconds') \
        .start()

print(f"Special Events Parquet stream query '{specials_query_name}' started successfully")

# Create a view of the special events data for joining in the foreachBatch
df_specials.writeStream \
    .format("memory") \
    .queryName("specials_table") \
    .outputMode("append") \
    .start()

# Generate user profile vectors - using a simpler approach without foreachBatch
# First prepare a stream with time components
df_rides_with_time = df_rides.select(
    "*",
    month("timestamp").alias("month"),
    dayofmonth("timestamp").alias("day"),
    hour("timestamp").alias("hour"),
    dayofweek("timestamp").alias("day_of_week"),
    dayofyear("timestamp").alias("day_of_year")
)

# Add watermark to the stream before creating the view
df_rides_with_time = df_rides_with_time.withWatermark("timestamp", "10 seconds")

def is_within_area(lat, lon, lat_min, lat_max, lon_min, lon_max):
    return (lat_min <= lat <= lat_max and 
            lon_min <= lon <= lon_max)

# Define UDF for determining ride relation to events
@udf(returnType=StringType())
def determine_event_relation(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, venue_lat, venue_lon):
    if (pickup_lat is None or pickup_lon is None or 
        dropoff_lat is None or dropoff_lon is None or 
        venue_lat is None or venue_lon is None):
        return "none"
    
    # Constants for latitude/longitude adjustments (0.5 km distance threshold)
    distance_km = 0.5
    latitude_adjustment = 0.009898 * distance_km
    longitude_adjustment = 0.00118 * distance_km

    # Define intervals around the venue
    lat_min = venue_lat - latitude_adjustment
    lat_max = venue_lat + latitude_adjustment
    lon_min = venue_lon - longitude_adjustment
    lon_max = venue_lon + longitude_adjustment

    # Check if pickup/dropoff locations are within the event area
    pickup_in_area = is_within_area(pickup_lat, pickup_lon, lat_min, lat_max, lon_min, lon_max)
    dropoff_in_area = is_within_area(dropoff_lat, dropoff_lon, lat_min, lat_max, lon_min, lon_max)

    if pickup_in_area and not dropoff_in_area:
        return "from_event"
    elif not pickup_in_area and dropoff_in_area:
        return "to_event"
    else:
        return "none"

# Create a memory view of the rides stream
df_rides_with_time.writeStream \
    .format("memory") \
    .queryName("rides_stream") \
    .outputMode("append") \
    .start()

# First, add the needed columns for event analysis
enriched_df = df_rides_with_time.withColumn(
    "is_weekend", 
    when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0)
)

# Initialize default values for event fields
enriched_df = enriched_df.withColumn("event_relation", lit("none").cast(StringType()))
enriched_df = enriched_df.withColumn("event_name", lit("none").cast(StringType()))

# Process special events if available
try:
    # Check if there are special events available from the specials_table memory view
    specials_df = spark.sql("SELECT * FROM specials_table")
    
    if not specials_df.rdd.isEmpty():
        # For each special event, apply the UDF to determine the ride relation to that event
        for event in specials_df.collect():
            # Extract event info
            event_name = event.event_name
            venue_lat = event.venue_latitude
            venue_lon = event.venue_longitude
            
            # Create a temporary column using the UDF.
            # Note: We pass constant venue values using lit()
            enriched_df = enriched_df.withColumn(
                "temp_relation",
                determine_event_relation(
                    col("pickup_latitude"), 
                    col("pickup_longitude"), 
                    col("dropoff_latitude"), 
                    col("dropoff_longitude"), 
                    lit(venue_lat), 
                    lit(venue_lon)
                )
            )
            
            # Update event_relation and event_name only for rows where the temp_relation is not "none"
            enriched_df = enriched_df.withColumn(
                "event_relation",
                when(col("temp_relation") != "none", col("temp_relation")).otherwise(col("event_relation"))
            )
            enriched_df = enriched_df.withColumn(
                "event_name",
                when(col("temp_relation") != "none", lit(event_name)).otherwise(col("event_name"))
            )
            
            # Remove the temporary column
            enriched_df = enriched_df.drop("temp_relation")
except Exception as e:
    print(f"Error processing special events: {e}")
    # Continue with default event relation values



enriched_df.writeStream \
    .format("memory") \
    .queryName("enriched_table") \
    .outputMode("append") \
    .start()

enriched_df.createOrReplaceTempView("enriched_table")

# Run a Spark SQL query to perform complex transformations
# (Here we aggregate metrics by user_id, but you can adapt the query as needed)
aggregated_df = spark.sql("""
    SELECT
        user_id,
        window(timestamp, "10 seconds") AS time_window,
        COUNT(*) AS total_rides,
        AVG(distance_km) AS avg_distance_km,
        MAX(distance_km) AS max_distance_km,
        AVG(pickup_latitude) AS avg_pickup_latitude,
        AVG(pickup_longitude) AS avg_pickup_longitude,
        AVG(dropoff_latitude) AS avg_dropoff_latitude,
        AVG(dropoff_longitude) AS avg_dropoff_longitude,
        CAST(SUM(CASE WHEN day_of_week IN (5, 6) THEN 1 ELSE 0 END) AS DOUBLE)/COUNT(*) AS weekend_ride_ratio,
        CAST(SUM(CASE WHEN event_relation = 'to_event' THEN 1 ELSE 0 END) AS DOUBLE)/COUNT(*) AS to_event_ratio,
        CAST(SUM(CASE WHEN event_relation = 'from_event' THEN 1 ELSE 0 END) AS DOUBLE)/COUNT(*) AS from_event_ratio,
        approx_count_distinct(event_name) AS unique_events_count,
        VARIANCE(distance_km) AS distance_variance,
        VARIANCE(hour) AS hour_variance
    FROM enriched_table
    GROUP BY user_id, window(timestamp, "10 seconds")
""")

# Write the aggregated result as a streaming query in append mode to Parquet
aggregated_query = aggregated_df.writeStream \
    .format("parquet") \
    .option("checkpointLocation", "checkpoint/parquet/aggregated") \
    .option("path", "output/aggregated") \
    .outputMode("append") \
    .trigger(processingTime='10 seconds') \
    .start()

print("Aggregated Query Status 1:", aggregated_query.status)


# Wait longer for parquet files to build up
print("Waiting for Parquet files to be created (60 seconds)...")
time.sleep(210)  # Increased from 20 to 60 seconds

print("Aggregated Query Status 2:", aggregated_query.status)

test_query = aggregated_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .trigger(processingTime='10 seconds') \
    .start()

# List output files
rides_output_files = os.listdir("output/rides")
specials_output_files = os.listdir("output/specials")
print(f"\nFiles in rides output directory: {len(rides_output_files)}")
if rides_output_files:
    print("Sample rides files:", rides_output_files[:5])
else:
    print("No rides output files created yet")

print(f"\nFiles in specials output directory: {len(specials_output_files)}")
if specials_output_files:
    print("Sample special events files:", specials_output_files[:5])
else:
    print("No special events output files created yet")

# Check for user vectors files too
try:
    user_vectors_output_files = os.listdir("output/user_vectors")
    print(f"\nFiles in user vectors output directory: {len(user_vectors_output_files)}")
    if user_vectors_output_files:
        print("Sample user vectors files:", user_vectors_output_files[:5])
    else:
        print("No user vectors output files created yet")
except Exception as e:
    print(f"Error checking user vectors directory: {e}")

# Display consolidated user vectors
print("\nChecking consolidated user vectors:")
try:
    if aggregated_query:
        consolidated_count = spark.sql("SELECT COUNT(*) as user_count FROM user_vectors_latest").collect()[0]['user_count']
        print(f"Number of users with profiles: {consolidated_count}")
        
        if consolidated_count > 0:
            print("\nSample user profiles:")
            spark.sql("""
                SELECT user_id, total_rides, avg_distance_km, max_distance_km, 
                      most_common_hour, most_common_day_of_week, 
                      weekend_ride_ratio, to_event_ratio, from_event_ratio,
                      most_common_event
                FROM user_vectors_latest
                LIMIT 5
            """).show(truncate=False)
            
            # Save a snapshot of the consolidated vectors to a file
            spark.sql("SELECT * FROM user_vectors_latest").write.mode("overwrite").parquet("output/consolidated_user_vectors")
            print("Saved consolidated user vectors to output/consolidated_user_vectors")
    else:
        print("No user vectors table exists yet. Waiting for data to be processed.")
except Exception as e:
    print(f"Error displaying consolidated user vectors: {e}")

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
    
    print("\nData saved to:")
    print(f"  - Rides data: {os.path.abspath('output/rides')}")
    print(f"  - Special events data: {os.path.abspath('output/specials')}")
    print(f"  - User profiles data: {os.path.abspath('output/user_vectors')}")

