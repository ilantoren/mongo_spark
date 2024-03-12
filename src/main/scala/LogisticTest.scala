import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.{ByteType, DoubleType, StringType, StructType}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

object LogisticTest extends App{
  val log = LogManager.getLogger("org.apache.logging.log4j.test2")
  log.setLevel(Level.INFO)

  log.info( "Start Logistic")
  val spark = SparkSession.builder()
    .master("local")
    .appName("Logistic")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt2.data")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt2.data")
    .getOrCreate()

  val aggregation =
    """
      [
      |  {
      |    '$match': {
      |      'key': {
      |        '$in': [
      |          3
      |        ]
      |      },
      |      'GoldsteinScale': {
      |        '$ne': null
      |      },
      |      'AvgTone': {
      |        '$ne': null
      |      },
      |      'Actor1CountryCode': {
      |        '$ne': null
      |      }
      |    }
      |  }, {
      |    '$addFields': {
      |      'label': {
      |        '$cond': {
      |          'if': {
      |            '$lte': [
      |              '$AvgTone', -2.092611
      |            ]
      |          },
      |          'then': 0,
      |          'else': 1
      |        }
      |      },
      |      'GoldsteinScale': {
      |        '$divide': [
      |          {
      |            '$add': [
      |              '$GoldsteinScale', 10
      |            ]
      |          }, 20
      |        ]
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
      |    '$project': {
      |      'label': 1,
      |      'key': 1,
      |      'GoldsteinScale': 1,
      |      'Actor1CountryCode': 1,
      |      'Actor1Code': 1,
      |      'Actor2Code': 1,
      |      'ActionGeo_CountryCode': 1,
      |      'AvgTone': 1
      |    }
      |  }
      |]""".stripMargin

  val schema = new StructType()
    .add( "_id", StringType, true)
    .add( "label", ByteType, true)
    .add( "key", ByteType, true)
    .add( "GoldsteinScale", DoubleType, true)
    .add( "Actor1CountryCode", StringType, true)
    .add( "Actor1Code", StringType, true)
    .add( "Actor2Code", StringType, true)
    .add( "ActionGeo_CountryCode", StringType, true)
    .add( "AvgTone", DoubleType, true)



  val data = spark
    .read
    .format("mongodb")
    .schema(schema)
    .options(Map("aggregation.pipeline" -> aggregation))
    .load()

  data.show()

  //data.createOrReplaceTempView("DATA")

  //val  splits: Array[Dataset[Row]]= spark.sql( "select *,  CAST( tone as TINYINT) as label from DATA ").randomSplit(Array(1,9))

  val splits = data.randomSplit(Array(1,9))
  val data2: Dataset[Row]= splits(1)

  val test = splits(0)


  val stringIndexer = new StringIndexer()
    .setInputCols( Array("ActionGeo_CountryCode", "Actor2Code", "Actor1Code", "Actor1CountryCode"))
    .setOutputCols( Array( "s1", "s2", "s3" , "s4"))
    .setHandleInvalid("skip")


  val encoder = new VectorAssembler().setInputCols( Array( "s1", "s2", "s3" , "s4", "GoldsteinScale")).setOutputCol( "vector")


  val scaler = new StandardScaler().setInputCol("vector").setOutputCol("features")

  // specify layers for the neural network:
  // input layer of size 4 (features), two intermediate of size 5 and 4
  // and output of size 3 (classes)
  val layers = Array[Int](5, 10, 5, 2)

  // create the trainer and set its parameters
  val trainer = new MultilayerPerceptronClassifier()
    .setLayers(layers)
    .setBlockSize(128)
    .setSeed(1234L)
    .setMaxIter(100)

  val pipeline = new Pipeline().setStages(Array(  stringIndexer,  encoder, scaler, trainer))


  val model = pipeline.fit(data2)

  model.transform(data2).show()


  val predictions = model.transform(test)

  predictions.printSchema()
  predictions.show()
  predictions.write.mode("overwrite").format( "parquet").save( "predictions")
 // predictions.select("_id", "label","prediction" ).write.format("mongodb").mode("overwrite")
 //  .options(Map("collection" -> "predictions"))
  //  .save()

  // Select (prediction, true label) and compute test error
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  log.info( "MultilayerPerceptronClassifier")
  log.info(s"Test set accuracy = $accuracy")

}
