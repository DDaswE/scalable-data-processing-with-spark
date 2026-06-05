"""
MIE1628 Assignment 2 - Part A-4
Word Frequency and Distribution Analysis

"""

import os
import sys
import re
from pyspark.sql import SparkSession

# Configure Spark Python environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


# Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, stddev, expr


# 1.Initialize Spark Session
spark = SparkSession.builder.appName("Word_Frequency").getOrCreate()

# 2. Load Data
file_path = "shakespeare-1.txt"
print(f"Loading data from: {file_path}")
print()

text_rdd = spark.sparkContext.textFile(file_path)
total_lines = text_rdd.count()
print(f"Loaded {total_lines:,} lines")
print()



# 3. MAPREDUCE FOR ALL WORDS

def process_partition_all_words(lines):
    import re
    token_pattern = re.compile(r'[a-z]+')
    local_counts = {}

    for line in lines:
        words = token_pattern.findall(line.lower())
        for word in words:
            local_counts[word] = local_counts.get(word, 0) + 1

    return local_counts.items()


# MapReduce pipeline to count ALL words
all_word_counts = (text_rdd
                   .mapPartitions(process_partition_all_words)
                   .reduceByKey(lambda a, b: a + b))

# Cache the result since we'll use it multiple times
all_word_counts.cache()

# Get basic statistics
total_words = all_word_counts.map(lambda x: x[1]).sum()
unique_words = all_word_counts.count()

print(f"Text Statistics:")
print(f"Total words:   {int(total_words):,}")
print(f"Unique words:  {unique_words:,}")
print()


# Top 10s
print("TOP 10 MOST FREQUENT WORDS")

# Sort by count descending and take top 10
top_10 = all_word_counts.takeOrdered(10, key=lambda x: -x[1])

print(f"{'Rank':<6} {'Word':<20} {'Count':>15} {'Percentage':>12}")
print("-" * 55)

for rank, (word, count) in enumerate(top_10, 1):
    percentage = (count / total_words) * 100
    print(f"{rank:<6} {word:<20} {count:>15,} {percentage:>11.4f}%")

print()

# Bottom 10s

print("BOTTOM 10 LEAST FREQUENT WORDS")

bottom_10 = all_word_counts.takeOrdered(10, key=lambda x: x[1])

print(f"{'Rank':<6} {'Word':<20} {'Count':>15} {'Percentage':>12}")
print("-" * 55)

for rank, (word, count) in enumerate(bottom_10, 1):
    percentage = (count / total_words) * 100
    print(f"{rank:<6} {word:<20} {count:>15,} {percentage:>11.4f}%")

print()


# DISTRIBUTION ANALYSIS

print("WORD FREQUENCY DISTRIBUTION ANALYSIS")

# collect
frequency_distribution = all_word_counts.map(lambda x: x[1]).collect()

# calcualte
from statistics import mean, median, stdev

freq_mean = mean(frequency_distribution)
freq_median = median(frequency_distribution)
freq_stdev = stdev(frequency_distribution) if len(frequency_distribution) > 1 else 0

print("Distribution Statistics:")
print(f"Mean frequency:       {freq_mean:.2f}")
print(f"Median frequency:     {freq_median:.2f}")
print(f"Std deviation:        {freq_stdev:.2f}")
print()


# Count words by frequency ranges
frequency_ranges = [
    (1, 1, "Exactly once (hapax legomena)"),
    (2, 5, "2-5 times"),
    (6, 10, "6-10 times"),
    (11, 50, "11-50 times"),
    (51, 100, "51-100 times"),
    (101, 500, "101-500 times"),
    (501, float('inf'), "500+ times")
]

print("Frequency Range Distribution:")
print(f"{'Range':<30} {'Word Count':>15} {'Percentage':>12}")
print("-" * 60)

for min_freq, max_freq, label in frequency_ranges:
    if max_freq == float('inf'):
        count = all_word_counts.filter(lambda x: x[1] >= min_freq).count()
    else:
        count = all_word_counts.filter(lambda x: min_freq <= x[1] <= max_freq).count()

    percentage = (count / unique_words) * 100
    print(f"{label:<30} {count:>15,} {percentage:>11.2f}%")

print()

# Visualization


try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    # Top 10 Most Frequent Words
    top_words = [word for word, count in top_10]
    top_counts = [count for word, count in top_10]

    plt.figure(figsize=(8, 6))
    bars = plt.barh(range(len(top_words)), top_counts, color='steelblue', alpha=0.8, edgecolor='black')
    plt.yticks(range(len(top_words)), top_words)
    plt.gca().invert_yaxis()
    plt.xlabel('Frequency', fontweight='bold')
    plt.title('Top 10 Most Frequent Words', fontsize=14, fontweight='bold', pad=12)
    plt.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, top_counts)):
        plt.text(count, i, f' {count:,}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('word_top10.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: word_top10.png")



except Exception as e:
    print(f"Visualization error: {e}")


spark.stop()
print("\n----- Complete -----\n")
