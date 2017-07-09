
import twitter
import requests
from pyspark.mllib.feature import HashingTF, IDF #Features engineering generate tf-idf matrox
from pyspark.mllib.regression import LabeledPoint #Object LabeledPoint to build the supervised dataframes
from pyspark.mllib.classification import NaiveBayes  #Naive bayes classifiers
from pyspark.streaming.kafka import KafkaUtils
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

#Create a local spark context and a streaming context
sc = SparkContext("local[2]","PySparkStreamingTest")
ssc = StreamingContext(sc,15)


#Initialize the kafka stream
kafkaStream = KafkaUtils.createStream(ssc, "localhost:2181", "spark-streaming-consumer", {"tweets": 1})
lines = kafkaStream.map(lambda x: x[1])
counts = lines.flatMap(lambda line: line.split(" ").map(lambda word: (word,1)).reduceByKey(lambda a, b: a+b))
counts.pprint()

#Set log level to ERROR
sc.setLogLevel("WARN")

#Create Dstream to localhost
#lines = ssc.socketTextStream("localhost",9999)
#words = lines.flatMap(lambda line: line.split(" "))
# Count each word in each batch
#pairs = words.map(lambda word: (word, 1))
#wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
#wordCounts.pprint()

#print lines
ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate
