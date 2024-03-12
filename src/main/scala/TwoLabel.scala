import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ByteType, DoubleType, StringType, StructType}

trait TwoLabel {
 def aggregation() =
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
     |     '_id': {'$toString': '$_id'},
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

  def schema() = new StructType()
    .add("_id", StringType, true)
    .add("label", ByteType, true)
    .add("key", ByteType, true)
    .add("GoldsteinScale", DoubleType, true)
    .add("Actor1CountryCode", StringType, true)
    .add("Actor1Code", StringType, true)
    .add("Actor2Code", StringType, true)
    .add("ActionGeo_CountryCode", StringType, true)
    .add("AvgTone", DoubleType, true)
}
