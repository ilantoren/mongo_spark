# Public Repository Technology Summary

This document summarizes the public repositories owned by [ilantoren](https://github.com/ilantoren), grouped and described by their primary technology.

## Scala

### [mongo_spark](https://github.com/ilantoren/mongo_spark)

A Scala and Apache Spark playground for analyzing GDELT event data stored in MongoDB. It uses the MongoDB Spark Connector, Spark SQL, and Spark MLlib for data import, aggregation, statistical summaries, histograms, linear regression, and cross-validation experiments. The repository also includes Python utilities using PyMongo and pandas for generating CSV and Parquet summaries.

### [-Akka-Throttle](https://github.com/ilantoren/-Akka-Throttle)

A Scala/Akka example that explores flow control and throttling for systems that produce messages faster than a downstream service can consume them. The implementation uses an in-memory queue to buffer work and release it more slowly, with the design intended to demonstrate how transient overload can be managed before scaling infrastructure.

### [akka-router-scala-spring](https://github.com/ilantoren/akka-router-scala-spring)

A Scala project demonstrating Akka routers, worker actors, lifecycle management, and Spring dependency injection. A collector distributes Fibonacci calculations across worker actors, then coordinates shutdown using router termination and actor monitoring. The injected business logic also provides a structure that can be tested independently from the actor system.

## Kotlin

### [UpdateOneField](https://github.com/ilantoren/UpdateOneField)

A Kotlin Coroutines project using the MongoDB coroutine driver to process and update large collections in batches. It focuses on partitioning GDELT data and applying aggregation-driven field changes while avoiding the need to load an entire collection into memory. The workflow is designed around MongoDB aggregation pipelines, batched reads, and batched updates for statistical analysis.

### [MongoHackathon2022](https://github.com/ilantoren/MongoHackathon2022)

A Kotlin-based GDELT data ingestion and analysis project integrating Google BigQuery, MongoDB Atlas, the Atlas Data API, Ktor, and Google Cloud services. It incrementally retrieves events from BigQuery, converts them into Kotlin data classes, uploads them to Atlas, and uses Atlas triggers/functions to reshape geographic and actor data. The repository also documents R and ggplot2 workflows for analyzing the resulting data.

### [stackOverflow](https://github.com/ilantoren/stackOverflow)

A small MongoDB-focused Kotlin repository documenting a Stack Overflow solution for round-robin selection. The approach uses `findOneAndUpdate`-style aggregation logic to select an array element and advance a counter with modular arithmetic. It also describes a two-collection variant that keeps a counter separate from the larger item collection for better scalability.

### [off-kotlin-example](https://github.com/ilantoren/off-kotlin-example)

A Kotlin example for extracting USDA links from an Open Food Facts-style products collection. It uses the MongoDB reactive driver together with Kotlin Coroutines to demonstrate asynchronous database access and processing. The project is part of the Mongo Loves Data examples.

## Rust

### [mongo-rust-async](https://github.com/ilantoren/mongo-rust-async)

An asynchronous Rust example for working with MongoDB product data. The project uses the MongoDB async driver and Tokio as its asynchronous runtime to extract USDA-related links from a products collection. It demonstrates a compact Rust/MongoDB integration pattern for non-blocking database operations.

## Dart and Flutter

### [mongo-loves-data-flutter](https://github.com/ilantoren/mongo-loves-data-flutter)

A Flutter application paired with an Express/Node.js server. The Flutter component uses Realm data and Google Maps to display restaurant information, while the server provides a simple HTTP interface for querying restaurants by neighborhood and cuisine. The mobile application requires a Google Maps API key and communicates with the accompanying Realm-backed service through environment-based configuration.

## JavaScript, Play Framework, AngularJS, and Solr

### [Play-with-Solr](https://github.com/ilantoren/Play-with-Solr)

A Play Framework 2.2 web application integrating AngularJS, MongoDB, and Solr. It demonstrates using Solr for autocomplete/search while retrieving the corresponding content from MongoDB through Play REST endpoints. Play's EHCache is used to cache previous suggestions and improve response performance.

## CSS / frontend troubleshooting

### [ui-angular-meteor-bug](https://github.com/ilantoren/ui-angular-meteor-bug)

A small public repository associated with an Angular and Meteor UI issue. GitHub identifies CSS as its primary language, but the repository currently contains only the default project documentation rather than a detailed technology guide. It appears to serve primarily as a reproducible frontend bug or integration example.

## Technology overview

Across these repositories, the main themes are MongoDB data access, asynchronous application design, and data analysis. The projects use Scala/Akka and Spark for distributed processing, Kotlin and Rust for typed asynchronous MongoDB applications, Flutter and JavaScript for user-facing applications, and Solr for search-oriented web experiences. GDELT and MongoDB appear repeatedly as the central data and analysis domain.
