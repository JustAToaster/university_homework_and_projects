import sys
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F

conf = SparkConf().setMaster("local[*]")
sc = SparkContext(conf=conf)
path1 = './instacart_2017_05_01/order_products__train.csv'
path2 = './instacart_2017_05_01/products.csv'
spark = SparkSession.builder.appName("Python").getOrCreate()
d1 = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(path1)
d2 = spark.read.options(header='True', inferSchema='True', delimiter=',').csv(path2)
df = d1.join(d2, ['product_id'])

basketdata = df.dropDuplicates(['order_id', 'product_id']).sort('order_id')
basketdata = basketdata.groupBy("order_id").agg(F.collect_list("product_name")).sort('order_id')
# df.repartition(1).write.format('com.databricks.spark.csv').save('mycsv.csv')

# df = spark.createDataFrame([(0, [1, 2, 5]),(1, [1, 2, 3, 5]),(2, [1, 2])], ["id", "items"])

print("Executing FPGrowth 1 with minSupport=0.01, minConfidence=0.5")
fpGrowth1 = FPGrowth(itemsCol="collect_list(product_name)", minSupport=0.01, minConfidence=0.5)
model = fpGrowth1.fit(basketdata)

# Display frequent itemsets.
model.freqItemsets.show()

print("There are " + str(model.freqItemsets.count()) + " frequent itemsets")

# Display generated association rules.
model.associationRules.show()

print("There are " + str(model.associationRules.count()) + " association rules")

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(basketdata).show()


print("Executing FPGrowth 2 with minSupport=0.001, minConfidence=0.5")
fpGrowth2 = FPGrowth(itemsCol="collect_list(product_name)", minSupport=0.001, minConfidence=0.5)
model = fpGrowth2.fit(basketdata)

# Display frequent itemsets.
model.freqItemsets.show()

print("There are " + str(model.freqItemsets.count()) + " frequent itemsets")

# Display generated association rules.
model.associationRules.show()

print("There are " + str(model.associationRules.count()) + " association rules")

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(basketdata).show()

sc.stop()