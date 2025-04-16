"""
Setup module for configuring Spark and environment
"""
import os
import subprocess
import time
import requests
import re
from packaging import version

def check_environment():
    """Check if running on Windows or Linux"""
    return os.name == 'nt'

def setup_linux_environment(spark_release):
    """Set up Spark environment on Linux"""
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
    
    return spark_home_dir

def setup_windows_environment(spark_release):
    """Set up Spark environment on Windows"""
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
    
    return spark_home_dir

def get_latest_spark_version():
    """Get the latest Spark version"""
    is_windows = check_environment()
    
    if not is_windows:
        # LINUX: Get the latest Spark version
        spark_version = subprocess.run(
            "curl -s https://downloads.apache.org/spark/ | grep -o 'spark-3\\.[0-9]\\+\\.[0-9]\\+' | sort -V | tail -1",
            shell=True, capture_output=True, text=True
        ).stdout.strip()
    else:
        # WINDOWS: Use Python requests to get the latest Spark version
        try:
            response = requests.get("https://downloads.apache.org/spark/")
            matches = re.findall(r'spark-3\.\d+\.\d+', response.text)
            if matches:
                sorted_versions = sorted(matches, key=lambda x: version.parse(x.replace('spark-', '')))
                spark_version = sorted_versions[-1] if sorted_versions else "spark-3.5.1"
            else:
                spark_version = "spark-3.5.1"  # Fallback version
        except Exception as e:
            print(f"Error fetching Spark version: {e}")
            spark_version = "spark-3.5.1"  # Fallback version
    
    print(f"Spark version: {spark_version}")
    return spark_version

def setup_environment():
    """Setup the environment for Spark"""
    is_windows = check_environment()
    spark_version = get_latest_spark_version()
    
    if is_windows:
        spark_home_dir = setup_windows_environment(spark_version)
    else:
        spark_home_dir = setup_linux_environment(spark_version)
    
    return spark_home_dir

def initialize_directories():
    """Initialize necessary directories for the application"""
    directories = [
        "checkpoint",
        "checkpoint/rides",
        "checkpoint/parquet",
        "checkpoint/parquet/rides",
        "checkpoint/parquet/specials",
        "checkpoint/parquet/user_vectors",
        "output",
        "output/rides",
        "output/specials",
        "output/user_vectors"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory {directory} created or already exists") 