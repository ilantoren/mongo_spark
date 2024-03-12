import breeze.stats.variance
import org.apache.log4j.{FileAppender, Level, LogManager}
import org.apache.spark.sql.{Row, SaveMode, SparkSession, functions}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.avg

object LinearRegressionExample extends App {
   val spark = SparkSession.builder()
     .master("local")
     .appName("LinearRegressionExample")
     .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt.data")
     .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt.data")
     .getOrCreate()

   val log = LogManager.getRootLogger


   val df = spark.read.format("mongodb").load()

   /**
    * Selection from mongodb based on the Spark api.  This is converted to a Mongodb aggregation
    *   and is influenced by indices on the collection,
    *   [{"$match":{"$and":[{"GoldsteinScale":{"$exists":true}},{"AvgTone":{"$exists":true}}]}},{"$group":{"_id":1,"n":{"$sum":1}}}]
    *   "planSummary":"IXSCAN { GoldsteinScale: 1 }","keysExamined":4400660,"docsExamined":4400660,
    */
   val data = df.select("AvgTone", "GoldsteinScale", "Actor1Code", "Actor1CountryCode", "EventCode")
     .filter( "GoldsteinScale is NOT NULL")
     .filter ( "AvgTone is NOT NULL")

   val summary = data.select( "AvgTone", "GoldsteinScale", "Actor1CountryCode")
     .groupBy("Actor1CountryCode")
     .agg(
        avg("AvgTone"), functions.count( "AvgTone"), functions.variance("AvgTone"),functions.stddev("AvgTone"),
        avg( "GoldsteinScale"), functions.variance("GoldsteinScale"), functions.stddev("GoldsteinScale")
     )

   data.show(50)


   summary.show(40)

   val actor = (r: Row) => {
      log.info("actor")
      log.info(r.getMap(1))
   }

   summary.write.mode(SaveMode.Overwrite).parquet("summary.parquet")


   /*
      In this example GoldsteinScale is a double
      and the regression is testing whether there is a
      linear relationship between AvgTone and the GoldsteinScale
    */
   val assembler = new VectorAssembler().setInputCols(Array( "GoldsteinScale"))
     .setOutputCol("features")
     .setHandleInvalid("skip")



   val lr = new LinearRegression()
     .setLabelCol("AvgTone")
     .setFeaturesCol("features")
     .setMaxIter(10)
     .setRegParam(0.3)
     .setElasticNetParam(0.8)




   val pipeline = new Pipeline().setStages(Array( assembler, lr))

   // Fit the model
   val pipelineModel = pipeline.fit(data)
   pipelineModel.write.overwrite().save("pipeline_model")
   val lrm: Option[LinearRegressionModel] = pipelineModel.stages.collectFirst { case t: LinearRegressionModel => t }

   lrm match {
      case Some(v) => {
         v.write.overwrite().save("lrModel_model")
         printOut(v)
      }
      case None => None
   }


/******************************  PART TWO **************/

val indexer = new StringIndexer()
  .setInputCol("Actor1CountryCode")
  .setOutputCol("ActorIndex")
  .setHandleInvalid("skip")

   val assembler2 = new VectorAssembler().setInputCols(Array("GoldsteinScale", "ActorIndex"))
     .setOutputCol("features")
     .setHandleInvalid("skip")
   val pipeline2 = new Pipeline().setStages(Array(indexer, assembler2, lr))

   // Fit the model
   val pipelineModel2 = pipeline2.fit(data)
   pipelineModel2.write.overwrite().save("pipeline2_model")
   val lrm2: Option[LinearRegressionModel] = pipelineModel2.stages.collectFirst { case t: LinearRegressionModel => t }

   lrm2 match {
      case Some(v) => {
         v.write.overwrite().save("lrModel2_model")
         printOut(v)
      }
      case None => None
   }


   // Print the coefficients and intercept for linear regression
   def printOut(lrModel: LinearRegressionModel) {
      log.info(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
      // Summarize the model over the training set and print out some metrics
      val trainingSummary = lrModel.summary

      log.info(s"numIterations: ${trainingSummary.totalIterations}")
      log.info(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
      trainingSummary.residuals.show()
      log.info(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
      log.info(s"r2: ${trainingSummary.r2}")
   }


}
