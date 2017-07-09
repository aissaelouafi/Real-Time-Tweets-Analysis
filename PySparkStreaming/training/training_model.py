import twitter
import requests
import datetime
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer #Features engineering generate tf-idf matrox
from pyspark.mllib.regression import LabeledPoint #Object LabeledPoint to build the supervised dataframes
from pyspark.mllib.classification import NaiveBayes  #Naive bayes classifiers
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import PCA
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType

sc = SparkContext()
sc.setLogLevel("ERROR")
#Intialize SparkSession, it will replace the SparkContext and the SQLContext in Spark 2.1.1
spark = SparkSession.builder.appName("Real Time Tweets Sentiment Analysis").getOrCreate()



#Define the schema of the csv train data
schema = StructType([
    StructField("ItemID", IntegerType()),
    StructField("Sentiment", IntegerType()),
    StructField("SentimentSource", StringType()),
    StructField("SentimentText",StringType())
])

#Upload the csv train data and show the first 20 lines of the dataframe
df = spark.read.csv("/Users/aissaelouafi/Desktop/MLProjets/RealTimeTweetsAnalysis/PySparkStreaming/data/train_data.csv",header=True,schema=schema)
print("\n The sentiment Text train data : ")
df.select("SentimentText").show()


#Filter the dataframe and show the positive tweets
labels = df.select("Sentiment")
print("\n The first positive tweets : ")
df.filter(df['Sentiment'] == 1).show()


#Tokenize sentiment text
tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SetimentTextTokenize")
wordsData = tokenizer.transform(df)

hashingTF = HashingTF(inputCol="SetimentTextTokenize", outputCol="rawFeatures", numFeatures=1000)
featurizedData = hashingTF.transform(wordsData)

#Show first element of the train data (Sentiment and rawFeatures columns)
featurizedData.select("Sentiment","rawFeatures").show

#Show the number of lines
print(featurizedData.count())

#Select only the first 10000 observations
featurizedData = featurizedData.filter(featurizedData.ItemID <= 500)


#Show the raw features column using the TF function
print("\n The raw features (TF) using hashing function : ")
featurizedData.select('rawFeatures').show()

#Convert the spark dataframe to RDD
train_rdd = featurizedData.select("Sentiment", "rawFeatures").rdd

#Convert the spark RDD to LabeledPoint object, this object will be the input of our NaiveBayes classifiers models
training = train_rdd.map(lambda x: LabeledPoint(x[0], x[1:]))

# Train the NaiveBayes classifier
model = NaiveBayes.train(training)



#labels_and_preds = labels.zip(model.predict(tf)).map(lambda x: {"actual": x[0], "predicted": float(x[1])})


#print labels_and_preds.collect()
print("\n Save the model : ")
current_timestamp = datetime.datetime.fromtimestamp(int("1284101485")).strftime('%Y-%m-%d-%H:%M:%S')


try:
    path = "./ML_models/NaiveClassifier/naiveBayesClassifier-"+current_timestamp
    model.save(sc,path)
    print("Model : "+path+" saved successfuly")
except:
    print "Unexcepted error ! Model unsaved"
