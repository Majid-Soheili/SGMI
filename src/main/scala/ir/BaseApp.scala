package ir

import java.io.FileOutputStream

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Bucketizer, QuantileDiscretizer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
 * The base class of spark applications
 * new update
 */
trait BaseApp extends Logging {

  final val LOCAL_FILE_PREFIX = "D://DataSets/"
  final val CLUSTER_FILE_PREFIX = "hdfs:///data/"
  final val SEED = 31
  var localExecution: Boolean = true

  //region --------- Create Sessions ----------------------------------------------------------------------

  def createSession(appName: String): SparkSession = if (localExecution) createStandaloneSession(appName) else createClusterSession(appName)

  def createStandaloneSession(appName: String, numberCores: Int = 5): SparkSession = {
    val session = SparkSession
      .builder()
      .appName(name = appName)
      .config("spark.master", s"local[$numberCores]")
      .config("spark.eventLog.enabled", value = true)
      .config("spark.driver.maxResultSize", "4g")
      .config("spark.eventLog.dir", "file:///D:/eventLogging/")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

    session.sparkContext.hadoopConfiguration.set("dfs.block.size", "128m")
    session
  }

  def createClusterSession(appName: String): SparkSession = {
    SparkSession
      .builder()
      .appName(name = appName)
      .getOrCreate()
  }

  //endregion

  //region --------- Read Data ----------------------------------------------------------------------------

  def readData(spark: SparkSession, s: DataSetSchema): DataFrame = if (s.isBinary) readBinaryData(spark, s) else readCsvData(spark, s)

  def readCsvData(spark: SparkSession, s: DataSetSchema): DataFrame = {

    val path = if (localExecution) LOCAL_FILE_PREFIX + s.fileName
    else CLUSTER_FILE_PREFIX + s.fileName

    spark.read.format(source = "csv")
      .option("delimiter", ",").option("quote", "")
      .option("header", "false")
      .schema(s.Schema).load(path)
  }

  def readBinaryData(spark: SparkSession, s: DataSetSchema): DataFrame = {
    val path = if (localExecution) LOCAL_FILE_PREFIX + s.fileName
    else CLUSTER_FILE_PREFIX + s.fileName

    val rdd = spark.sparkContext.binaryRecords(path, s.binaryRecordLength)
      .map(Row.fromSeq(_))
    spark.createDataFrame(rdd, s.Schema)
  }

  //endregion

  //region --------- Discretization -----------------------------------------------------------------------

  def SimpleDiscretizer(data: DataFrame, nBuckets: Int, s: DataSetSchema): DataFrame = {

    val continuous = s.cofNames.toArray
    val continuousDisc = continuous.map(c => s"${c}_disc")
    val label = s.cName

    val discretizer = new QuantileDiscretizer()
      .setInputCols(continuous)
      .setOutputCols(continuousDisc)
      .setNumBuckets(nBuckets)

    val assembler = new VectorAssembler()
      .setInputCols(continuousDisc)
      .setOutputCol("FeaturesVector")
    val indexer = new StringIndexer()
      .setInputCol(label)
      .setOutputCol("labelIndex")
    val pipeline = new Pipeline()
      .setStages(Array(discretizer, assembler, indexer))
    val model = pipeline.fit(data)

    val newNames = Seq("features", label)
    model.transform(data).select("FeaturesVector", "labelIndex").toDF(newNames: _*)
  }

  def WriteSimpleDiscretizer(data: DataFrame, nBuckets: Int, s: DataSetSchema):Unit = {

    val path = if (localExecution)
      LOCAL_FILE_PREFIX + "parquet/" + s.name + ".parquet"
    else
      CLUSTER_FILE_PREFIX + "parquet/" + s.name + ".parquet"

    val continuous = s.cofNames.toArray
    val continuousDisc = continuous.map(c => s"${c}_disc")

    val discretizer = new QuantileDiscretizer()
      .setInputCols(continuous)
      .setOutputCols(continuousDisc)
      .setNumBuckets(nBuckets)

    discretizer.fit(data).write.save(path)
  }

  def LoadSimpleDiscretizer(data: DataFrame, s: DataSetSchema): DataFrame = {

    val path = if (localExecution)
      LOCAL_FILE_PREFIX + "parquet/" + s.name + ".parquet"
    else
      CLUSTER_FILE_PREFIX + "parquet/" + s.name + ".parquet"

    val continuous = s.cofNames.toArray
    val continuousDisc = continuous.map(c => s"${c}_disc")
    val label = s.cName

    val bucketizer = Bucketizer.read.load(path)

    val assembler = new VectorAssembler()
      .setInputCols(continuousDisc)
      .setOutputCol("FeaturesVector")
    val indexer = new StringIndexer()
      .setInputCol(label)
      .setOutputCol("labelIndex")
    val pipeline = new Pipeline()
      .setStages(Array(bucketizer, assembler, indexer))
    val model = pipeline.fit(data)

    val newNames = Seq("features", label)
    model.transform(data).select("FeaturesVector", "labelIndex").toDF(newNames: _*)
  }

  //endregion

  //region --------- Saving ---------------------------------------------------

  def SaveOutput(spark: SparkSession, fileName: String, data: Array[Byte]): Unit = {

    if (localExecution)
      WriteInFS(fileName, data)
    else
      WriteInHDFS(spark, fileName, data)
  }

  def WriteInHDFS(spark: SparkSession, fileName: String, data: Array[Byte]): Unit = {

    val path = new Path(s"hdfs:///output/$fileName.csv")
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val os = fs.create(path)

    os.write(data)
    os.close()
    fs.close()
  }

  def WriteInFS(fileName: String, data: Array[Byte]): Unit = {
    val out = new FileOutputStream(LOCAL_FILE_PREFIX + "Output/" + fileName)
    out.write(data)
    out.close()
  }

  //endregion

}
