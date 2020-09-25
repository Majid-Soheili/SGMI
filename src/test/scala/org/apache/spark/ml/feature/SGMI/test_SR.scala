package org.apache.spark.ml.feature.SGMI

import org.apache.spark.ml.feature.{BasicSparkTest, DataSetSchema, VectorAssembler}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class test_SR extends BasicSparkTest {

  test(testName = "SGMI_SR-Synthetic") {

    val spark = createSession("SGMI_SR-synthetic")
    val schema = DataSetSchema.Synthetic
    val data = readDataFrame(spark, schema)
    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")
    val processedData = featureAssembler.transform(data)

    val model = new FeatureSelector()
      .setOptMethod("SR")
      .setMaxBin(10)
      .setBatchSize(1.0)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setUseCatch(false)
      .fit(processedData)

    val actualRank = model.getRank
    val expectedRank = Array(10, 11, 9, 8, 14, 15, 5, 16, 12, 4, 6, 13, 7, 3, 17, 18, 2, 19, 0, 1, 20)
    assert(actualRank === expectedRank)
  }

  test(testName = "SGMI-SR-Musk") {

    val spark = createSession("SGMI-SR-Musk")
    val schema = DataSetSchema.Musk
    val data = readDataFrame(spark, schema)
    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")
    val processedData = featureAssembler.transform(data)

    val model = new FeatureSelector()
      .setOptMethod("SR")
      .setMaxBin(30)
      .setBatchSize(0.3)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setUseCatch(false)
      .fit(processedData)

    val actualRank = model.getRank.take(10)
    val expectedRank = Array(161, 101, 164, 162, 35, 41, 91, 160, 165, 130)
    //val expectedRank = Array(125, 144, 109, 94, 66, 137, 150, 35, 162, 74)

    assert(actualRank === expectedRank)
  }

  test(testName = "SGMI-SR-Alpha") {

    val spark = createSession("SGMI-SR-Alpha")
    val schema = DataSetSchema.Alpha
    val data = readDataFrame(spark, schema)
    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")
    val processedData = featureAssembler.transform(data)

    val model = new FeatureSelector()
      .setOptMethod("SR")
      .setMaxBin(30)
      .setBatchSize(1.0)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setUseCatch(false)
      .fit(processedData)

    val actualRank = model.getRank.take(10)
    val expectedRank = Array(493, 285, 470, 498, 457, 297, 87, 114, 6, 427)

    assert(actualRank === expectedRank)
  }
}
