# mongo_spark

Examples and experiments for analyzing [GDELT](https://www.gdeltproject.org/) event data with Apache Spark, MongoDB, and Spark MLlib.

The project contains Scala applications for importing tab-delimited GDELT exports into MongoDB, querying and transforming the data with the MongoDB Spark Connector, calculating descriptive statistics and distributions, and fitting machine-learning models such as linear regression and cross-validated pipelines.

> **Status:** This is a personal data-analysis playground rather than a packaged library or production-ready application. Several examples contain local MongoDB URIs and filesystem paths that should be customized before running.

## Technology stack

- Scala 2.13.10
- Apache Spark 3.4.0
  - Spark Core
  - Spark SQL
  - Spark MLlib
  - Spark Streaming
- MongoDB Spark Connector 10.1.1
- SBT
- Python utilities using `pymongo`, `pandas`, and Parquet output

## Repository layout

| Path | Description |
| --- | --- |
| `src/main/scala/` | Scala Spark examples and applications |
| `src/main/resources/` | Spark logging configuration |
| `python/` | MongoDB aggregation and summary utilities |
| `sample_data/` | Sample GDELT export data |
| `chi_case*.json` | Example model/configuration artifacts |
| `*_distribution/` | Saved histogram data in Parquet format |
| `*_model/` | Saved Spark ML pipeline/model artifacts |
| `build.sbt` | SBT project definition and dependencies |

## Prerequisites

1. Install a JDK compatible with Spark 3.4.
2. Install [SBT](https://www.scala-sbt.org/).
3. Install and start MongoDB locally.
4. Obtain GDELT event data, or use the sample file in `sample_data/`.
5. For the Python utility, install the required packages:

   ```bash
   python -m pip install pymongo pandas pyarrow
   ```

## Configure MongoDB

The Scala examples currently use local MongoDB connection strings such as:

```text
mongodb://127.0.0.1/gdelt.data
```

The importer uses a different database/collection configuration by default. Review each example and update the `spark.mongodb.read.connection.uri` and `spark.mongodb.write.connection.uri` settings for your environment before running it.

The Python summary script expects the following local MongoDB layout:

```text
mongodb://localhost:27017/gdelt
collection: data
```

## Build the project

From the repository root:

```bash
sbt compile
```

To run one of the Scala applications locally, use SBT's `runMain` command. For example:

```bash
sbt 'runMain ScaleHistogram'
```

Other useful entry points include:

```bash
sbt 'runMain ImportIntoMongodb'
sbt 'runMain LinearRegressionExample'
sbt 'runMain FitWithoutUSA'
sbt 'runMain CrossValidationExample'
sbt 'runMain ToneScaleDistribution'
```

Generated Spark outputs are written to directories such as `goldstein_scale_distribution`, `avgtone_distribution`, `summary.parquet`, and the various saved model directories. These paths are relative to the working directory unless an example specifies an absolute path.

## Import GDELT data into MongoDB

`ImportIntoMongodb` defines a schema for GDELT event exports, reads tab-delimited files with Spark, and writes the resulting DataFrame to MongoDB through the MongoDB Spark Connector.

Before running it, update the input path in `src/main/scala/ImportIntoMongodb.scala`:

```scala
val path = "sample_data"
```

The importer expects GDELT-style tab-delimited records and includes fields such as:

- `GlobalEventId`
- actor and country codes
- event codes
- `GoldsteinScale`
- `AvgTone`
- geographic fields
- `DATEADDED`
- `SOURCEURL`

## Analysis examples

- **`ScaleHistogram`** — Reads `GoldsteinScale` from MongoDB and saves a histogram as Parquet.
- **`ToneScaleDistribution`** — Summarizes and creates distributions for `AvgTone` and `GoldsteinScale`.
- **`LinearRegressionExample`** — Computes country-level summaries and fits linear-regression models predicting `AvgTone` from `GoldsteinScale`, optionally including actor-country features.
- **`FitWithoutUSA`** — Fits a regression model after excluding records whose actor country is `USA`.
- **`CrossValidationExample`** — Builds a feature-hashing and linear-regression pipeline, searches a parameter grid with cross-validation, and reports RMSE, explained variance, and R².
- **`ChiSquareExample`**, **`LogisticTest`**, **`RandomForest`**, and **`TwoLabel`** — Additional Spark ML/statistical experiments.

## Python summary utility

`python/summary_gdelt.py` runs a MongoDB aggregation that groups records by `key` and calculates counts plus summary statistics for `AvgTone` and `GoldsteinScale`:

```bash
python python/summary_gdelt.py
```

It writes:

- `gdelt_summary.csv`
- `gdelt_summary_stats` (a Parquet dataset)

Ensure MongoDB is running and that the `gdelt.data` collection contains the expected GDELT fields before running the script.

## Notes and limitations

- The examples use `master` as the default branch and are configured for local execution.
- MongoDB connection URIs, database names, collection names, and data paths are embedded in several source files.
- Some examples assume fields such as `AvgTone`, `GoldsteinScale`, `Actor1CountryCode`, and `key` already exist in MongoDB.
- Saved model and output directories may already exist; Spark is generally configured to overwrite them in the examples.
- No automated test suite or production deployment configuration is currently included.

## License

No license has been specified for this repository. Unless a license is added, default copyright law applies and reuse should be treated accordingly.
