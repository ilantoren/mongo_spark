import org.apache.log4j.{FileAppender, Level, LogManager}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import java.io.FileWriter



object ChiSquareExample extends App {
  val spark = SparkSession.builder()
    .master("local")
    .appName("MongoSparkConnectorIntro")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .getOrCreate()

  val log = LogManager.getRootLogger
/*  log.setLevel(Level.INFO)
  val appender = new FileAppender()
  appender.setFile("chisquare.log", true, true, 8192)
  log.addAppender(appender)*/

  val aggregation =
    """
     [
      |  {
      |    '$match': {
      |      'Actor1CountryCode': {
      |        '$ne': null
      |      },
      |      'GoldsteinScale': {
      |        '$ne': null
      |      },
      |      'key': {
      |         '$in': [ 3, 7, 11]
      |      }
      |    }
      |  }, {
      |    '$addFields': {
      |      'Actor1Code': {
      |        '$ifNull': [
      |          '$Actor1Code', '$Actor1CountryCode'
      |        ]
      |      },
      |      'GoldsteinScale': {
      |        '$toString': {
      |          '$round': [
      |            '$GoldsteinScale'
      |          ]
      |        }
      |      },
      |      'scale': '$GoldsteinScale',
      |      'EventCode': {
      |        '$toString': '$EventCode'
      |      },
      |      'is_same': {
      |        '$cond': [
      |          {
      |            '$eq': [
      |              '$Actor1Code', '$Actor1CountryCode'
      |            ]
      |          }, true, false
      |        ]
      |      },
      |      'ToneLevel': {
      |        '$switch': {
      |          'branches': [
      |            {
      |              'case': {
      |                '$gte': [
      |                  '$AvgTone', -2.0569480243800276
      |                ]
      |              },
      |              'then': 'positive'
      |            }, {
      |              'case': {
      |                '$lt': [
      |                  '$AvgTone', -2.0569480243800276
      |                ]
      |              },
      |              'then': 'negative'
      |            }
      |          ],
      |          'default': 'invalid'
      |        }
      |      }
      |    }
      |  }, {
      |    '$project': {
      |      'AvgTone': 1,
      |      'scale': 1,
      |      'is_same': 1,
      |      'ToneLevel': 1,
      |      'GoldsteinScale': 1,
      |      'Actor1Code': 1,
      |      'Actor1CountryCode': 1,
      |      'EventCode': 1
      |    }
      |  }
      |]
      |""".stripMargin

  val data = spark.read.format("mongodb")
    .options(Map("aggregation.pipeline" -> aggregation))
    .load()

  data.withColumn("GoldsteinScale", col("GoldsteinScale").cast(StringType))
  data.show()

  // Basic case ToneLevel vs GoldsteinScale
  val indexer = new StringIndexer().setInputCol("GoldsteinScale").setOutputCol("scaleIndex").setHandleInvalid("keep")
  val stringIndexer = new StringIndexer().setInputCol("ToneLevel").setOutputCol("label")
  val vectorized = new VectorAssembler().setInputCols(Array("scaleIndex")).setOutputCol("features").setHandleInvalid("keep")
  val pipeline = new Pipeline().setStages(Array(indexer, stringIndexer, vectorized))
  val df2 = pipeline.fit(data).transform(data)
  runTests( "ToneLevel to GoldsteinScale","case1", df2 )

  // In addition to the GoldsteinScale consider the Actor1CountryCode
  val countryCodeIndexer = new StringIndexer().setInputCol("Actor1CountryCode").setOutputCol("CountryFeature").setHandleInvalid("keep")
  val vectorized2 = new VectorAssembler().setInputCols(Array("scaleIndex", "CountryFeature")).setOutputCol("features").setHandleInvalid("keep")
  val pipeline2 = new Pipeline().setStages(Array(indexer, countryCodeIndexer, stringIndexer, vectorized2))
  val df3 = pipeline2.fit(data).transform(data)

  runTests( "ToneLevel to GoldsteinScale and Actor1CountryCode","case2", df3)

  // Add in consideration for the is_same boolean feature
  val vectorized3 = new VectorAssembler().setInputCols(Array("is_same", "scaleIndex", "CountryFeature")).setOutputCol("features").setHandleInvalid("keep")

  val pipeline3 = new Pipeline().setStages(Array(indexer, countryCodeIndexer, stringIndexer, vectorized3))
  val df4 = pipeline3.fit(data).transform(data)
  runTests( "ToneLevel vs is_same, GoldsteinScale and Actor1CountryCode","case3",  df4)


  val linearRegression = new LinearRegression()
    .setLabelCol("AvgTone")
    .setFeaturesCol("features")
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)

  val linearRegressionModel = linearRegression.fit(df4)
  linearRegression.write.overwrite().save("chs_linearRegression")

  println( linearRegressionModel.summary )
  log.info( linearRegressionModel.summary )
  val ts = linearRegressionModel.summary
  log.info( s"R2 ${ts.r2}")
  log.info( s"Explained variance:  ${ts.explainedVariance}")
  log.info( s"Degrees of Freedom:  ${ts.degreesOfFreedom}")


  def runTests(message: String, name: String, dataFrame: DataFrame): Unit = {
    log.info(s"What is the relationship between  AvgTone and GoldsteinScale\n  $message \n")
    dataFrame.show()
    dataFrame.write.mode("overwrite").format("parquet").save( s"dataFrame_$name")
    val chi: Row = ChiSquareTest.test(dataFrame, "features", "label").head

    val fw = new FileWriter( s"chi_$name.json")
    fw.write(chi.prettyJson)
    fw.close()

    log.info(s"pValues = ${chi.getAs[Vector](0)}")
    log.info(s"degreesOfFreedom ${chi.getSeq[Integer](1).mkString("[", ",", "]")}")
    log.info(s"statistics ${chi.getAs[Vector](2)}")

    /** ANY LINEAR  RELATION BETWEEN  SCALE AND TONE */

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFeaturesCol("features")
      .setLabelCol("label")

    val lrModel = lr.fit(dataFrame)
    lrModel.write.overwrite().save(s"lrModel_$name")

    val trainingSummary = lrModel.summary

    // Obtain the objective per iteration
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(println)

    // for multiclass, we can inspect metrics on a per-label basis
    log.info("False positive rate by label:")
    trainingSummary.falsePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
      log.info(s"label $label: $rate")
    }

    log.info("True positive rate by label:")
    trainingSummary.truePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
      log.info(s"label $label: $rate")
    }

    log.info("Precision by label:")
    trainingSummary.precisionByLabel.zipWithIndex.foreach { case (prec, label) =>
      println(s"label $label: $prec")
    }

    log.info("Recall by label:")
    trainingSummary.recallByLabel.zipWithIndex.foreach { case (rec, label) =>
      log.info(s"label $label: $rec")
    }


    log.info("F-measure by label:")
    trainingSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
      log.info(s"label $label: $f")
    }

    val accuracy = trainingSummary.accuracy
    val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    val truePositiveRate = trainingSummary.weightedTruePositiveRate
    val fMeasure = trainingSummary.weightedFMeasure
    val precision = trainingSummary.weightedPrecision
    val recall = trainingSummary.weightedRecall
    log.info(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
      s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")
  }
}
/**
 * {"t":{"$date":"2023-03-05T21:58:43.143+02:00"},"s":"I",  "c":"COMMAND",  "id":51803,   "ctx":"conn2985","msg":"Slow query","attr":{"type":"command","ns":"gdelt.data","command":{"getMore":5177032722283437669,"collection":"data","$db":"gdelt","lsid":{"id":{"$uuid":"92a71e0e-23b5-4ef1-b630-31994a56afb1"}}},"originatingCommand":{"aggregate":"data","pipeline":[{"$match":{"_id":{"$gte":{"$oid":"63ba86bf309b986ac00d49bf"}}}},{"$match":{"$and":[{"GoldsteinScale":{"$exists":true}},{"AvgTone":{"$exists":true}}]}},{"$project":{"Actor1Code":1,"Actor1CountryCode":1,"AvgTone":1,"EventCode":1,"GoldsteinScale":1}}],"cursor":{},"allowDiskUse":true,"$db":"gdelt","lsid":{"id":{"$uuid":"92a71e0e-23b5-4ef1-b630-31994a56afb1"}}},"planSummary":"IXSCAN { _id: 1 }","cursorid":5177032722283437669,"keysExamined":1926,"docsExamined":1926,"fromMultiPlanner":true,"cursorExhausted":true,"numYields":3,"nreturned":19
 */