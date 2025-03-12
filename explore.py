import pandas as pd
from fastavro import reader

# Specify the path to your Avro file
file_path = "C:/Users/fabia/Downloads/Group project/data/ride_events.avro"

# Read the Avro file and convert it to a Pandas DataFrame
avro_records = []
with open(file_path, 'rb') as fo:
    avro_reader = reader(fo)
    for record in avro_reader:
        avro_records.append(record)

# Create a DataFrame from the list of records
df = pd.DataFrame(avro_records)


# Display the DataFrame
print(df)


