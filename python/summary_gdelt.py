from pymongo import MongoClient
import pandas as pd

client = MongoClient('mongodb://localhost:27017/')
result = client['gdelt']['data'].aggregate([
    {
        '$match': {
            'Actor1CountryCode': {
                '$exists': 0
            }, 
            'Actor1CountryCode': {
                '$exists': 1
            },
            'GoldsteinScale': {
                '$exists': 1
            },
            'AvgTone': {
                '$exists': 1
            }
        }
    }, {
        '$group': {
            '_id': '$key',
            'count': {
                '$sum': 1
            }, 
            'avg_tone': {
                '$avg': '$AvgTone'
            }, 
            'sd_tone': {
                '$stdDevPop': '$AvgTone'
            }, 
            'avg_scale': {
                '$avg': '$GoldsteinScale'
            }, 
            'sd_scale': {
                '$stdDevPop': '$GoldsteinScale'
            }
        }
    }
], allowDiskUse=True)

df = pd.DataFrame(result)
df.to_parquet('gdelt_summary_stats')
df.to_csv( "gdelt_summary.csv")