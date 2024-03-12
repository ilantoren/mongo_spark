import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructType}

object ImportIntoMongodb extends App {
  val spark = SparkSession.builder()
    .master("local")
    .appName("ImportGdeltToMongo")
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/gdelt2.data")
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/gdelt2.data")
    .getOrCreate()

  println("First SparkContext:")
  println("APP Name :" + spark.sparkContext.appName)
  println("Deploy Mode :" + spark.sparkContext.deployMode)
  println("Master :" + spark.sparkContext.master)


  val schema = new StructType().add("GlobalEventId", IntegerType, true)
    .add("Day", IntegerType, true)
    .add("MonthYear", IntegerType, true)
    .add("Year", IntegerType, true)
    .add("FractionDate", DoubleType, true)
    .add("Actor1Code", StringType, true)
    .add("Actor1Name", StringType, true)
    .add("Actor1CountryCode", StringType, true)
    .add("Actor1KnownGroupCode", StringType, true)
    .add("Actor1EthnicCode", StringType, true)
    .add("Actor1Religion1Code", StringType, true)
    .add("Actor1Religion2Code", StringType, true)
    .add("Actor1Type1Code", StringType, true)
    .add("Actor1Type2Code", StringType, true)
    .add("Actor1Type3Code", StringType, true)
    .add("Actor2Code", StringType, true)
    .add("Actor2Name", StringType, true)
    .add("Actor2CountryCode", StringType, true)
    .add("Actor2KnownGroupCode", StringType, true)
    .add("Actor2EthnicCode", StringType, true)
    .add("Actor2Religion1Code", StringType, true)
    .add("Actor2Religion2Code", StringType, true)
    .add("Actor2Type1Code", StringType, true)
    .add("Actor2Type2Code", StringType, true)
    .add("Actor2Type3Code", StringType, true)
    .add("IsRootEvent", IntegerType, true)
    .add("EventCode", StringType, true)
    .add("EventBaseCode", StringType, true)
    .add("EventRootCode", StringType, true)
    .add("QuadClass", IntegerType, true)
    .add("GoldsteinScale", DoubleType, true)
    .add("NumMentions", IntegerType, true)
    .add("NumSources", IntegerType, true)
    .add("NumArticles", IntegerType, true)
    .add("AvgTone", DoubleType, true)
    .add("Actor1Geo_Type", IntegerType, true)
    .add("Actor1Geo_Fullname", StringType, true)
    .add("Actor1Geo_CountryCode", StringType, true)
    .add("Actor1Geo_ADM1Code", StringType, true)
    .add("Actor1Geo_ADM2Code", StringType, true)
    .add("Actor1Geo_Lat", StringType, true)
    .add("Actor1Geo_Long", StringType, true)
    .add("Actor1Geo_FeatureID", StringType, true)
    .add("Actor2Geo_Type", IntegerType, true)
    .add("Actor2Geo_Fullname", StringType, true)
    .add("Actor2Geo_CountryCode", StringType, true)
    .add("Actor2Geo_ADM1Code", StringType, true)
    .add("Actor2Geo_ADM2Code", StringType, true)
    .add("Actor2Geo_Lat", StringType, true)
    .add("Actor2Geo_Long", StringType, true)
    .add("Actor2Geo_FeatureID", StringType, true)
    .add("ActionGeo_Type", IntegerType, true)
    .add("ActionGeo_Fullname", StringType, true)
    .add("ActionGeo_CountryCode", StringType, nullable = true)
    .add("ActionGeo_ADM1Code", StringType, true)
    .add("ActionGeo_ADM2Code", StringType, true)
    .add("ActionGeo_Lat", StringType, true)
    .add("ActionGeo_Long", StringType, true)
    .add("ActionGeo_FeatureID", StringType, true)
    .add("DATEADDED", StringType, true)
    .add("SOURCEURL", StringType, true)

  // val path = "/Volumes/archive-drive/gdelt/main_data"
  val path = "/Volumes/archive-drive/gdelt3/main_data"
 //val path = "sample_data"
  val df = spark.read
    .format("csv")
    .schema(schema)
    .options(Map("delimiter" -> "\t")).load(path)

  df.show()

  df.write.format("mongodb").mode("overwrite")
    .options(Map("collection" -> "gdelt2"))
    .save()

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
      |      }
      |    }
      |  }, {
      |    '$addFields': {
      |      'Actor1Code': {
      |        '$ifNull': [
      |          '$Actor1Code', '$Actor1CountryCode'
      |        ]
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
      |                  '$AvgTone', 0.5035
      |                ]
      |              },
      |              'then': 'positive'
      |            }, {
      |              'case': {
      |                '$lte': [
      |                  '$AvgTone', -4.4850
      |                ]
      |              },
      |              'then': 'negative'
      |            }
      |          ],
      |          'default': 'neutral'
      |        }
      |      }
      |    }
      |  }, {
      |    '$project': {
      |      'AvgTone': 1,
      |      "_id": 1,
      |      'scale': 1,
      |      'is_same': 1,
      |      'ToneLevel': 1,
      |      'GoldsteinScale': 1,
      |      'Actor1Code': 1,
      |      'Actor1CountryCode': 1,
      |      'EventCode': 1
      |    }
      |  }, {
      |    '$limit': 50000
      |  }
      |]
      |""".stripMargin

  /** Use the aggregation pipeline to read in data.
   * then save the results for future use
   */
  val df2 = spark.read.format("mongodb")
    .options(Map("aggregation.pipeline" -> aggregation)).load()
  df2.show()
  val summary = df2.summary("count", "mean", "stddev", "25%", "75%")
  println( "SUMMARY")
  summary.show(100)
  df2.write.mode("overwrite").format("mongodb")
    .options(Map("collection" -> "transformed_data"))
    .save()

  summary.drop("_id").write.mode("overwrite").format("mongodb")
    .options(Map( "collection" -> "sample_summary"))
    .save()

  /**
   *  SUMMARY
   *  23/03/04 22:46:50 INFO CodeGenerator: Code generated in 15.37074 ms
   *  +-------+----------+-----------------+-------------------+------------------+------------------+---------+-------+------------------+
   *  |summary|Actor1Code|Actor1CountryCode|            AvgTone|         EventCode|    GoldsteinScale|ToneLevel|    _id|             scale|
   *  +-------+----------+-----------------+-------------------+------------------+------------------+---------+-------+------------------+
   *  |  count|   2509911|          2509911|            2509911|           2509911|           2509911|  2509911|2509911|           2509911|
   *  |   mean|      null|             null|-2.0581127820778216| 98.13638690774295|0.5790102517579548|     null|   null|0.5790102517579548|
   *  | stddev|      null|             null|  3.854071308963629|189.30552421577025| 4.746730847358039|     null|   null| 4.746730847358039|
   *  |    25%|      null|             null|  -4.48504983388704|              36.0|              -2.0|     null|   null|              -2.0|
   *  |    75%|      null|             null|   0.50352467270897|             111.0|               3.4|     null|   null|               3.4|
   *  +-------+----------+-----------------+-------------------+------------------+------------------+---------+-------+------------------+
   *
   */
}
