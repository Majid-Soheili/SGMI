package ir

import org.apache.spark.ml.feature.SGMI.FeatureSelector

/**
 * Demo
 * Experimental Study
 * Change, Method name to one of (QP, SR, TP)
 * Change, Dataset
 * Current is FD dataset which is a binary file
 */
object Demo extends BaseApp {

  def main(args: Array[String]): Unit = {

    localExecution = args.length == 0
    val (method, schema, numberPartitions) = if (localExecution) {
      ("QP", DataSetSchema.Epsilon, 1.toShort)
    }
    else {
      val mtdName = args(0)
      val dsName = args(1)
      val numberPartitions: Short = if (args.length >= 3) args(2).toShort else 1
      val schema = if (dsName.toLowerCase.contains("ocr")) DataSetSchema.OCR
      else if (dsName.toLowerCase.contains("epsilon")) DataSetSchema.Epsilon
      else if (dsName.toLowerCase.contains("fd")) DataSetSchema.FD
      else if (dsName.toLowerCase.contains("ecbdl")) DataSetSchema.ECBDL
      else DataSetSchema.Alpha

      (mtdName, schema, numberPartitions)
    }

    val start: Long = System.currentTimeMillis()
    val appName = s"SGMI-$method-${schema.name}"
    val spark = super.createSession(appName)

    try {

      val train = if (numberPartitions == 1)
        super.readData(spark, schema)
      else
        super.readData(spark, schema).repartition(numberPartitions)

      val discTrain = super.LoadSimpleDiscretizer(train, schema)

      println(discTrain.count())

      val model = new FeatureSelector()
        .setOptMethod(method)
        .setMaxBin(12)
        .setBatchSize(0.25)
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setUseCatch(true)
        .fit(discTrain)

      val rank = model.getRank
      logInfo(s"Total computing time: ${System.currentTimeMillis() - start}")
      val outString = rank.mkString(",")
      super.SaveOutput(spark, appName, outString.getBytes)

    } finally {
      spark.close()
    }
  }
}
