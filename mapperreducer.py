"""
MIE1628 Assignment 2 - Part A-3
Implement an Optimized MapReduce Word Count

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


# Initialize Spark Session
spark = SparkSession.builder.appName("MapReduce_Word_Count").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

#  Load Data
file_path = "shakespeare-1.txt"
print(f"Loading data from: {file_path}")
print()



# CONFIGURATION
target_words = ["Shakespeare", "What", "The", "Lord", "Library",
                "GUTENBERG", "WILLIAM", "COLLEGE", "WORLD"]

print("Target Words to Count:")
for word in target_words:
    print(f"  {word}")
print()


#OPTIMIZATION 1: Broadcast Variable

target_words_lower = [w.lower() for w in target_words]
target_set_broadcast = spark.sparkContext.broadcast(set(target_words_lower))
print(f"Broadcasted {len(target_words_lower)} target words to all workers\n")

# LOAD DATA
text_rdd = spark.sparkContext.textFile(file_path)
total_lines = text_rdd.count()
print(f"Loaded {total_lines:,} lines")
print(f"Sample (first 3 lines):")
for i, line in enumerate(text_rdd.take(3), 1):
    preview = line[:80] + "..." if len(line) > 80 else line
    print(f"  {i}. {preview}")
print()
#OPTIMIZATION 2: mapPartitions
def count_words_in_partition(lines):

    import re
    token_pattern = re.compile(r'[a-z]+')

    target_set = target_set_broadcast.value
    local_counts = {}

    for line in lines:
        words = token_pattern.findall(line.lower())

        for word in words:
            if word in target_set:
                local_counts[word] = local_counts.get(word, 0) + 1


    return local_counts.items()


#MAPREDUCE PIPELINE
word_counts = (text_rdd
               .mapPartitions(count_words_in_partition)
               .reduceByKey(lambda a, b: a + b))

results = word_counts.collect()



print("\nRESULTS: Word Frequency Count")


word_display = {w.lower(): w for w in target_words}
results_sorted = sorted(results, key=lambda x: (-x[1], x[0]))

print(f"{'Word':<20} {'Count':>15}")
print("-" * 35)

total_count = 0
for word_lower, count in results_sorted:
    display_word = word_display.get(word_lower, word_lower.capitalize())
    print(f"{display_word:<20} {count:>15,}")
    total_count += count

print("-" * 35)
print(f"{'TOTAL':<20} {total_count:>15,}")
print()

# ===== ANALYSIS =====
found_words = {w for w, c in results}
missing_words = set(target_words_lower) - found_words

if missing_words:
    print("Words NOT found in the text:")
    for w in sorted(missing_words):
        print(f"{word_display.get(w, w.capitalize())}")
else:
    print("All target words were found!")

if results_sorted:
    most = results_sorted[0]
    least = results_sorted[-1]
    print(f"\nMost common:  {word_display[most[0]]} ({most[1]:,} times)")
    print(f"Least common: {word_display[least[0]]} ({least[1]:,} times)")

print()

# Cleanup
target_set_broadcast.unpersist()
spark.stop()

print("Analysis Complete!\n")
