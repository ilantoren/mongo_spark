import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{PCA, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

object RandomForest extends App{
  val log = LogManager.getLogger("org.apache.logging.log4j.test2")
  log.setLevel(Level.INFO)

  val spark = SparkSession.builder()
    .master("local")
    .appName("RandomForest")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt.data")
    .getOrCreate()

  def prepData(): Dataset[Row] = {

    val aggregation =
      """
        |[
        |  {
        |    '$match': {
        |       'GoldsteinScale' : {
        |         '$ne': null
        |       },
        |      'key': {
        |        '$in': [
        |          3,4,5,6,7,8,9,10
        |        ]
        |      }
        |    }
        |  }, {
        |    '$addFields': {
        |      'tone': {
        |        '$switch': {
        |          'branches': [
        |            {
        |              'case': {
        |                '$lt': [
        |                  '$AvgTone', -5.3
        |                ]
        |              },
        |              'then': 1
        |            }, {
        |              'case': {
        |                '$lt': [
        |                  '$AvgTone', -3
        |                ]
        |              },
        |              'then': 2
        |            }, {
        |              'case': {
        |                '$lt': [
        |                  '$AvgTone', -1
        |                ]
        |              },
        |              'then': 3
        |            }, {
        |              'case': {
        |                '$lt': [
        |                  '$AvgTone', 1.4
        |                ]
        |              },
        |              'then': 4
        |            }, {
        |              'case': {
        |                '$lt': [
        |                  '$AvgTone', 11
        |                ]
        |              },
        |              'then': 5
        |            }
        |          ],
        |          'default': -1
        |        }
        |      },
        |      'Actor1Code': {
        |        '$ifNull': [
        |          '$Actor1Code', '$Actor1CountryCode'
        |        ]
        |      },
        |      'Actor2Code': {
        |        '$ifNull': [
        |          '$Actor2Code', ''
        |        ]
        |      }
        |    }
        |  }, {
        |    '$match': {
        |      'tone': {
        |        '$ne': 0
        |      },
        |      'Actor1CountryCode': {
        |        '$ne': null
        |      }
        |    }
        |  }, {
        |    '$project': {
        |      'tone': 1,
        |      'key': 1,
        |      'GoldsteinScale': {'$divide': [{'$add':[  '$GoldsteinScale', 10]}, 20]},
        |      'Actor1CountryCode': 1,
        |      'Actor1Code': 1,
        |      'Actor2Code': 1,
        |      'ActionGeo_CountryCode': 1
        |    }
        |  },
        |
        |
        |]""".stripMargin


    val data = spark.read.format("mongodb").options(Map("aggregation.pipeline" -> aggregation)).load()
    return data
  }
  val data = prepData()
  data.createOrReplaceTempView("DATA")

  val  splits: Array[Dataset[Row]]= spark.sql( "select *,  CAST( tone as TINYINT) as label from DATA where tone > 0 ").randomSplit(Array(1,9))
  val data2: Dataset[Row]= splits(1)

  val test = splits(0)


  val stringIndexer = new StringIndexer()
    .setInputCols( Array("ActionGeo_CountryCode", "Actor2Code", "Actor1Code", "Actor1CountryCode"))
    .setOutputCols( Array( "s1", "s2", "s3" , "s4"))
    .setHandleInvalid("skip")

  //val scaler = new StandardScaler().setInputCol( "GoldsteinScale").setOutputCol("scale")

  val encoder = new VectorAssembler().setInputCols( Array( "s1", "s2", "s3" , "s4", "GoldsteinScale")).setOutputCol( "vectored")

  val pca = new PCA().setInputCol("vectored").setOutputCol("features").setK(5)

  val indexed = new VectorIndexer().setInputCol( "tone").setOutputCol( "label")

  val randomForest = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(6)

  val pipeline = new Pipeline().setStages(Array(  stringIndexer,  encoder, pca , randomForest))


  val model = pipeline.fit(data2)

  model.transform(data2).show()


  val predictions = model.transform(test)

  predictions.show()

  // Select (prediction, true label) and compute test error
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println(s"Test set accuracy = $accuracy")
  log.info(s"Test set accuracy = $accuracy")

}
