import org.apache.spark.sql.SparkSession


object ToneScaleDistribution  extends App {
  val spark = SparkSession.builder()
    .master("local")
    .appName("BasicStatsScale")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt2.gdelt2")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt2.gdelt2")
    .getOrCreate()

  /**
   * Read from MongoDB and create a Temp View table for the Spark SQL
   */
   spark.read.format("mongodb").load().createOrReplaceTempView("gdelt")
   spark.sqlContext.cacheTable("gdelt")
  val df1 =  spark.sqlContext.sql("SELECT AvgTone, GoldsteinScale from gdelt where AvgTone is not null and GoldsteinScale is not null")
  df1.show()

  /* Generate a Summary of the two columns */
  val summary = df1.summary()
  summary.show()

  /* Take the DataFrame and convert it to a RDD for Histogram
  *    In order to plot both the GoldsteinScale and AvgTone on the same
  *    x-axis specify the bins to use
  * */
  val rdd = df1.select("AvgTone").toJavaRDD.mapToDouble(num => num.getDouble(0))
  val range: Array[Double]  = Range(-120,128).map( _/8.toDouble).toArray

  // Calculate the histogram counts for each bin
  val histogram:  Array[Long] = rdd.histogram( range )

  //  Zip to bins and save to a parquet file
  val data = range.zip(histogram)
  val df2 = spark.createDataFrame(data)
  df2.write.mode("overwrite").format("parquet").save("avgtone_distribution")

 // DO the same for the GoldsteinScale
  val rdd2 = df1.select("GoldsteinScale").toJavaRDD.mapToDouble(num => num.getDouble(0))
  val histogram2: Array[Long] = rdd2.histogram(range)
  val data2 = range.zip(histogram2)
  val df3 = spark.createDataFrame(data2)
  df3.write.mode("overwrite").format("parquet").save("goldstein_scale_distribution")

}
