import LinearRegressionExample.printOut
import org.apache.spark.sql.{Row, SaveMode, SparkSession, functions}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.functions.{avg, col}
import org.apache.spark.sql.types.IntegerType


object FitWithoutUSA extends App {
  val spark = SparkSession.builder()
    .master("local")
    .appName("MongoSparkConnectorIntro")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .getOrCreate()

  val df = spark.read.format("mongodb").load()

  val data = df.select("AvgTone", "GoldsteinScale", "Actor1Code", "Actor1CountryCode")
    .filter( "Actor1CountryCode is NOT NULL")
    .filter("GoldsteinScale is NOT NULL")
    .filter("AvgTone is NOT NULL")
    .filter( df("Actor1CountryCode").=!=("USA"))



  val assembler = new VectorAssembler().setInputCols(Array("GoldsteinScale"))
    .setOutputCol("features")
    .setHandleInvalid("skip")


  val lr = new LinearRegression()
    .setLabelCol("AvgTone")
    .setFeaturesCol("features")
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)


  val pipeline = new Pipeline().setStages(Array(assembler, lr))

  // Fit the model
  val pipelineModel = pipeline.fit(data)
  pipelineModel.write.overwrite().save("pipeline_model")
  val lrm: Option[LinearRegressionModel] = pipelineModel.stages.collectFirst { case t: LinearRegressionModel => t }

  lrm match {
    case Some(v) => {
      v.write.overwrite().save("absent_USA")
      printOut(v)
    }
    case None => None
  }

}
