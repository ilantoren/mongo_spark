ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"
val sparkVersion = "3.4.0"

lazy val root = (project in file("."))
  .settings(
    name := "mongo_spark"
  )
libraryDependencies ++= Seq(
  "org.mongodb.spark" %% "mongo-spark-connector" % "10.1.1",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
)
