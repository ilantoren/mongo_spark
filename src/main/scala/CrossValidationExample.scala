import EvaluateFit.{explained_var, r2, rmse}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.FeatureHasher
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object CrossValidationExample extends App{

  val log = LogManager.getLogger("org.apache.logging.log4j.test2")
  log.setLevel(Level.INFO)

  val spark = SparkSession.builder()
    .master("local")
    .appName("CrossValidationExample")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .getOrCreate()

  val aggregation =
    """
[
      {
        '$match': {
          'key': {
            '$in': [
              2, 20, 6
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
  data.show(300)
  val keys = Seq( 2, 20)
  val training = data.filter(col("key").isin( keys:_*))
  val test = data.filter( col("key") === 6)

  val countryIndex = new FeatureHasher()
    .setInputCols(Array("Actor1Code",  "Actor1CountryCode","ActionGeo_CountryCode", "GoldsteinScale"))
    .setOutputCol("features")


  val lr = new LinearRegression()
    .setLabelCol("AvgTone")
    .setFeaturesCol("features")
    .setMaxIter(10)

  val pipeline = new Pipeline().setStages(Array(countryIndex,  lr))
  // We use a ParamGridBuilder to construct a grid of parameters to search over.
  // TrainValidationSplit will try all combinations of values and determine best model using
  // the evaluator.
  val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(0, 0.1, 0.3, 0.5))
    .addGrid(lr.fitIntercept)
    .addGrid(lr.elasticNetParam, Array(0,  0.5, 1.0))
    .addGrid( countryIndex.numFeatures, Array(  1000, 3000, 5000))
    .build()


  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new RegressionEvaluator())
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(5) // Use 3+ in practice
    .setParallelism(3) // Evaluate up to 2 parameter settings in parallel

  // In this case the estimator is simply the linear regression.
  // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
/*  val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(lr)
    .setEvaluator(new RegressionEvaluator)
    .setEstimatorParamMaps(paramGrid)
    // 80% of the data will be used for training and the remaining 20% for validation.
    .setTrainRatio(0.8)
    // Evaluate up to 2 parameter settings in parallel
    .setParallelism(2)*/

  // Run train validation split, and choose the best set of parameters.
  val cvModel = cv.fit(training)
  cvModel.transform(test).show(400)
  val desc = cvModel.bestModel.extractParamMap()
  log.info( s"ParamMap: $desc")
  cvModel.write.overwrite().save("crossValidation")





  // Make predictions on test data. model is the model with combination of parameters
  // that performed best.

 val set =  cvModel.bestModel.transform(test)
    .select( "label", "prediction")

  // Calculate RMSE
  val regressionEvaluator: RegressionEvaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
  val rmse = regressionEvaluator.evaluate(set)

  // Calculate the explained variance
  regressionEvaluator.setMetricName("var")
  val explained_var = regressionEvaluator.evaluate(set)

  // Calculate the R2
  regressionEvaluator.setMetricName("r2")
  val r2 = regressionEvaluator.evaluate(set)

  println(s"METRIC: ${rmse}, Explained variance: ${explained_var},   R2: ${r2}")

}
