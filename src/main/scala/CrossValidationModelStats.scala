import CrossValidationExample.spark
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object CrossValidationModelStats extends  App {
  val spark = SparkSession.builder()
    .master("local")
    .appName("CrossValidationModelStats")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .getOrCreate()

  val log = LogManager.getLogger("org.apache.logging.log4j.test2")
  log.setLevel(Level.INFO)
  log.info( "Starting the application")
  val aggregation =
    """
[
      {
        '$match': {
          'key': {
            '$in': [
              2, 20, 6,  9, 10, 11, 12, 13
            ]
          },
          'Actor1CountryCode': {
            '$ne': null
          }
        }
      }, {
        '$project': {
          'key': 1,
          'Actor1Code': {
            '$ifNull': [
              '$Actor1Code', '$Actor1CountryCode'
            ]
          },
          'Actor1CountryCode': 1,
          'Actor2CountryCode': {
            '$ifNull': [
              '$Actor2CountryCode', '$Actor1CountryCode'
            ]
          },
          'ActionGeo_CountryCode': {
            '$ifNull': [
              '$ActionGeo_CountryCode', '$Actor1CountryCode'
            ]
          },
          'GoldsteinScale': {
            '$divide': [
              {
                '$add': [
                  '$GoldsteinScale', 10
                ]
              }, 20
            ]
          },
          'AvgTone': 1,
          'label': {
            '$divide': [
              {
                '$add': [
                  '$AvgTone', 20
                ]
              }, 40
            ]
          }
        }
      }
    ]
  """
  val data = spark.read.format("mongodb").options(Map("aggregation.pipeline" -> aggregation)).load()
  //val training = data.filter(col("key") < 6)
  val test = data.filter(col("key") === 6)
  log.info( "Showing the data")
  test.show(100)
  log.info( "Load the saved model: crossValidation")
  val savedModel = CrossValidatorModel.load("crossValidation")
  val pipelineModel = savedModel.bestModel
  log.info( "use the model to predict from the current data")
  val fittedData = pipelineModel.transform(data)
  fittedData.show(500)

  fittedData.write.mode("overwrite").format("parquet").save("fitted_data")


  val set = fittedData.select(col("label"), col("prediction"))
  set.show()
  // Calculate RMSE
  val regressionEvaluator: RegressionEvaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
  val rmse = regressionEvaluator.evaluate(set)

  // Calculate the explained variance
  regressionEvaluator.setMetricName("var")
  val explained_var = regressionEvaluator.evaluate(set)

  // Calculate the R2
  regressionEvaluator.setMetricName("r2")
  val r2 = regressionEvaluator.evaluate(set)
  val params = pipelineModel.extractParamMap().toSeq
  params.foreach(println(_))

}
