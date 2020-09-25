package org.apache.spark.ml.feature

import org.apache.spark.ml.feature.DataSetSchema.baseSchema
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class BasicSparkTest extends BasicTest {

  def createSession(name:String): SparkSession = {
    def spark: SparkSession = SparkSession
      .builder()
      .appName(name = name)
      .config("spark.master", "local[6]")
      .getOrCreate()

    spark
  }

  def readRDD(spark: SparkSession, s: baseSchema): RDD[Row] = {

    val path = "src/test/scala/resources/data/" + s.name + ".csv"
    val context = spark.sparkContext
    val rdd = context.textFile(path)

    val cIndex = s.cIndex
    rdd.map(line => line.split(",")).map(line => Row.fromSeq(for (i <- 0 to cIndex) yield line(i).trim.toByte))
  }

  def readDataFrame(spark: SparkSession, s: baseSchema): DataFrame = {

    val path = "src/test/scala/resources/data/" + s.name + ".csv"
    spark.read.format("csv")
      .option("delimiter", ",").option("quote", "")
      .option("header", "false")
      .schema(s.Schema).load(path)
  }
}
