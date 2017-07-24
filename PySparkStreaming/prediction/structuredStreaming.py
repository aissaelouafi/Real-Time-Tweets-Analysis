from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.ml import Pipeline, PipelineModel



spark = SparkSession.builder.appName("StructuredNetworkWordCount").getOrCreate()


# Create DataFrame representing the stream of input lines from connection to localhost:9999
lines = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()

# Split the lines into words
test = lines.select(
   explode(
       split(lines.value, "\s{2,}")
   ).alias("text")
)
# Generate running word count
model_saved = PipelineModel.load("/tmp/pipeline/")

prediction = model_saved.transform(test)

selected = prediction.select("text", "probability", "prediction")



# Start running the query that prints the running counts to the console
query = selected \
    .writeStream \
    .format("console") \
    .start()


query.awaitTermination()
