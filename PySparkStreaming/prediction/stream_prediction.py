import twitter
import requests
import sys
import re
import json
from pyspark.mllib.feature import HashingTF, IDF #Features engineering generate tf-idf matrox
from pyspark.mllib.regression import LabeledPoint #Object LabeledPoint to build the supervised dataframes
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel  #Naive bayes classifiers
from pyspark.streaming.kafka import KafkaUtils
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import HashingTF, Tokenizer #Features engineering generate tf-idf matrox
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType

from pyspark.sql import Row, SQLContext



def predict(rdd):
    count = rdd.count()
    if(count>0):
        hashingTF = HashingTF(numFeatures=1000)
        features = hashingTF.transform(rdd.map(lambda doc:doc["text"].split(" "))).
        result = model.transform(features)
        print result.probability
        #Here write result to SOLR
    else:
        print("No data receveid")


#Create a local spark context and a streaming context
sc = SparkContext("local[3]","PySparkStreamingTest")
ssc = StreamingContext(sc,15)
spark = SparkSession.builder.appName("Real Time Tweets Sentiment Analysis").getOrCreate()

#Set Log Level to ERROR
sc.setLogLevel("ERROR")


#Import the Naive bayes classifiers model
model = NaiveBayesModel.load(sc, "ML_models/NaiveClassifier/naiveBayesClassifier-2010-09-10-08-51-25")

#Initialize the kafka stream
lines = ssc.socketTextStream("localhost", 9999)

#kafkaStream = KafkaUtils.createStream(ssc, "localhost:2181", "spark-streaming-consumer", {"tweets": 1})
#weets  = lines.flatMap(lambda tweet:tweet).flatMap(lambda tweet:json.loads(tweet)).map(lambda tweet: (tweet.get('text')))


#Define the schema of the local context dataframe
schema = StructType([
    StructField("SentimentText", StringType())
])


#Parse string to json object
tweets = lines.map(lambda v: json.loads(v))

#Extract the text field from the json object
text_dstream = tweets.map(lambda tweet: tweet['text'])
id_dstream = tweets.map(lambda tweet: tweet['id'])
#text_dstream.pprint()


#Goup by the id
id_count_groupBy = id_dstream.countByValue()
id_count_groupBy.pprint()

#Count tweet by text
author_counts = text_dstream.countByValue()
author_counts.pprint()


df = text_dstream.foreachRDD(lambda rdd: predict(rdd))



ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate


#Convert the dstream to dataframes object ==> so here we have a local context all stream receveid will be in the dataframes
#df = text_dstream.transform(lambda rdd: sc.parallelize(rdd))
#hashingTF = HashingTF()
#predictions = text_dstream.foreachRDD(lambda rdd: predict(rdd))


#Apply the features hashing to the local df (we should apply the same features hashing as the train data )
#tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SetimentTextTokenize")
#wordsData = tokenizer.transform(df)

#hashingTF = HashingTF(inputCol="SetimentTextTokenize", outputCol="rawFeatures", numFeatures=1000)
#featurizedData = hashingTF.transform(wordsData)

#Show the raw features of the local context df
#featurizedData.select("rawFeatures").show()

#Convert the test datafrale to a RDD
#test_rdd = featurizedData.select("rawFeatures").rdd

#Convert the spark RDD to LabeledPoint object, this objesdsddct will be the input of our NaiveBayes classifiers models
# test = test_rdd.map(lambda p: (model.predict(p.rawFeatures)))
#test.pprint
