import twitter
import requests
import datetime
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF, IDF #Features engineering generate tf-idf matrox
from pyspark.mllib.regression import LabeledPoint #Object LabeledPoint to build the supervised dataframes
from pyspark.mllib.classification import NaiveBayes  #Naive bayes classifiers
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import PCA


sc = SparkContext("local[2]","PySparkStreamingTest")
sc.setLogLevel("ERROR")

#Functions
def g(x):
    print x


#Define an obsolete train dataset
train_data = sc.parallelize([{"text":"Aissa EL OUAFI je mappelle","label":0.0},
                            {"text":"Messi est le meilleur joueur du monde","label":1.0},
                            {"text":"Messi est un joueur magnifique ...","label":1.0},
                            {"text":"Ronaldo est tellement nul","label":1.0}
                            ])


#train_data.map(lambda doc: doc["text"].replace("...",""))
#Split the train data in test and train
labels = train_data.map(lambda doc:doc["label"])
print("Label : ********************")
print labels.collect()


tf = HashingTF(numFeatures=1000).transform( ## Use much larger number in practice
    train_data.map(lambda doc: doc["text"].split(" "),
    preservesPartitioning=True))

#idf = IDF().fit(tf)
#tfidf = idf.transform(tf)
print("\nTF : ******************** ")
print tf.collect()


print("\nRT TF : ******************** ")
test_2 = HashingTF(numFeatures=1000).transform("Bonjour je suis Aissa tres intelligent et vous ?!!".split(" "))
print test_2


# Combine using zip
training = labels.zip(tf).map(lambda x: LabeledPoint(x[0], x[1]))


print("\nTraining data : ******************** ")
print training.collect()


# Train and check
model = NaiveBayes.train(training)
labels_and_preds = labels.zip(model.predict(tf)).map(lambda x: {"actual": x[0], "predicted": float(x[1])})


print labels_and_preds.collect()
#current_timestamp = datetime.datetime.fromtimestamp(int("1284101485")).strftime('%Y-%m-%d-%H:%M:%S')
#print(current_timestamp)
#model.save(sc,"/rf/"+current_timestamp)
#print("Model saved successfuly")
