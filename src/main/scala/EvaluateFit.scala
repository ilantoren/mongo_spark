import org.apache.log4j.LogManager
import org.apache.log4j.Level
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object EvaluateFit extends App {
  val spark = SparkSession.builder()
    .master("local")
    .appName("EvaluateFit")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .getOrCreate()

  val log = LogManager.getLogger("org.apache.logging.log4j.test2")
  log.setLevel(Level.INFO)
  log.info("Starting the application")

  val dataset = spark.read.format("parquet").load("fitted_data")
  val set = dataset.select(col("label"), col("prediction"))
  set.show()
  // Calculate RMSE
  val regressionEvaluator: RegressionEvaluator = new RegressionEvaluator( ).setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
  val rmse = regressionEvaluator.evaluate(set)

  // Calculate the explained variance
  regressionEvaluator.setMetricName("var")
  val explained_var = regressionEvaluator.evaluate(set)

  // Calculate the R2
  regressionEvaluator.setMetricName("r2")
  val r2 = regressionEvaluator.evaluate(set)

  println( s"METRIC: ${rmse}, Explained variance: ${explained_var},   R2: ${r2}")
}
