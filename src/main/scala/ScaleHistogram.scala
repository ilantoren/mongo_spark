import org.apache.spark.sql.SparkSession


object ScaleHistogram extends App {
  val spark = SparkSession.builder()
    .master("local")
    .appName("ImportGdeltToMongo")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .getOrCreate()

  val pipeline = """
  [ { "$match": { "$and": [
  { "GoldsteinScale": { "$exists": 1 } },
  { "GoldsteinScale": { "$ne": "" } },
  { "GoldsteinScale": { "$ne": null } } ] } },
  { "$project": { "GoldsteinScale": 1,  "_id": 0 } }]
  """

  val df = spark.read.format("mongodb").options(Map( "aggregation.pipeline" -> pipeline)).load()

  df.show()

  val rdd = df.select( "GoldsteinScale").toJavaRDD.mapToDouble(num => num.getDouble(0) )

  val range: Array[Double] = Range(-20, 42).map(_/2.toDouble).toArray

  val histogram: Array[Long] = rdd.histogram(range)

  val data = range.zip(histogram)

  val df2 = spark.createDataFrame(data)

  df2.show()

  df2.write.mode("overwrite").format("parquet").save("goldstein_scale_distribution")

}
