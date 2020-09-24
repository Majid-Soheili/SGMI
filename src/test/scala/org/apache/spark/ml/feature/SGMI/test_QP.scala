package org.apache.spark.ml.feature.SGMI

import org.apache.spark.ml.feature.{BasicSparkTest, DataSetSchema, VectorAssembler}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class test_QP extends BasicSparkTest {

  test(testName = "SGMI-QP-Musk") {

    val spark = createSession("SGMI-QP-Musk")
    val schema = DataSetSchema.Musk
    val data = readDataFrame(spark, schema)
    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")
    val processedData = featureAssembler.transform(data)

    val model = new FeatureSelector()
      .setOptMethod("QP")
      .setMaxBin(50)
      .setBatchSize(0.25)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setUseCatch(false)
      .fit(processedData)

    val actualRank = model.getRank.take(10)
    val expectedRank = Array(125, 144, 109, 94, 66, 137, 150, 35, 162, 74)

    assert(actualRank === expectedRank)
  }

  test(testName = "SGMI-QP-Alpha") {

    val spark = createSession("SGMI-QP-Alpha")
    val schema = DataSetSchema.Alpha
    val data = readDataFrame(spark, schema)
    val featureAssembler = new VectorAssembler()
      .setInputCols(schema.fNames.toArray)
      .setOutputCol("features")
    val processedData = featureAssembler.transform(data)

    val model = new FeatureSelector()
      .setOptMethod("QP")
      .setMaxBin(50)
      .setBatchSize(1.0)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setUseCatch(false)
      .fit(processedData)

    val actualRank = model.getRank.take(10)
    val expectedRank = Array(468, 498, 493, 87, 29, 331, 465, 357, 360, 346)

    assert(actualRank === expectedRank)
  }
}
