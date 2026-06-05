"""
MIE1628 Assignment 2 - Part A-1
Count Odd and Even Numbers
"""

import os
import sys
import builtins


# Configure Spark Python environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, when


# instantiate the spark session
spark = SparkSession.builder.appName("OddEvenCount").getOrCreate()


# Get data
integerRDD = spark.sparkContext.textFile("integer.txt")

# Convert a str to int
nums = integerRDD.map(lambda x: int(x.strip()))

# mark the even or odd numbers
kvRDD = nums.map(lambda x: ("even", 1) if x % 2 == 0 else ("odd", 1))

# Aggregate
countsRDD = kvRDD.reduceByKey(lambda x, y: x + y)

# Action
results = countsRDD.collect()


for category, count in results:
    print(f"{category}: {count}")

total = sum([count for _, count in results])
print(f"Total_Number: {total}")
print()

# Stop spark session
spark.stop()
print("Yes! Program completed successfully!")
