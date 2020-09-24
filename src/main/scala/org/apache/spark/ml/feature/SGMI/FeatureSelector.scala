package org.apache.spark.ml.feature.SGMI

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.feature.Model.DFSModel
import org.apache.spark.ml.feature.OptimizationMethods.{QP, SR, TP}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.StructType
import org.apache.spark.storage.StorageLevel

/**
  *
  */
final class FeatureSelector (override val uid: String) extends Estimator[DFSModel] with HasFeaturesCol with HasLabelCol {

  def this() = this(Identifiable.randomUID("SGMI-FS"))

  // -------- Parameters ------------------------------------------

  val OptMethod:Param[String] = new Param[String](this, "OptMethod", "The optimization method applied for ranking features", ParamValidators.inArray(Array("QP", "SR", "TP")))
  val maxBin: Param[Int] = new Param[Int](this, "maxBin", "The maximum bins of discretized data set")
  val batchSize: Param[Double] = new Param[Double](this, "batchSize", "The batch size of computing mutual information")
  val useCatch: Param[Boolean] = new Param[Boolean](this, "useCatch", "Using the catch")

  setDefault(useCatch, false)
  setDefault(OptMethod, "QP")

  def getMaxBin: Int = $(maxBin)

  def getBatchSize: Double = $(batchSize)

  def getUseCatch: Boolean = $(useCatch)

  def setMaxBin(num: Int): this.type = set(maxBin, num)

  def setBatchSize(num: Double): this.type = set(batchSize, num)

  def setFeaturesCol(name: String): this.type = set(featuresCol, name)

  def setLabelCol(name: String): this.type = set(labelCol, name)

  def setUseCatch(value: Boolean): this.type = set(useCatch, value)


  // -------- Overriding Functions --------------------------------

  override def fit(data: Dataset[_]): DFSModel = {

    val rdd: RDD[Row] = data.select(this.getLabelCol, this.getFeaturesCol).rdd
    val nFeatures = rdd.first().getAs[Any](this.getFeaturesCol) match {
      case v: DenseVector => v.size
      case v: SparseVector => v.size
    }
    val nColumns = nFeatures + 1

    val (train, nInstances) = if (this.getUseCatch) {
      val t = rdd.mapPartitions(partition => transformColumnar(partition, nColumns)).persist(StorageLevel.MEMORY_AND_DISK)
      val n = t.mapPartitions(p => Iterator(p.next().length)).sum().toInt
      (t, n)
    }
    else {
      val n = data.count().toInt
      val t = rdd.mapPartitions(partition => transformColumnar(partition, nColumns))
      (t, n)
    }


    val weights = Array.emptyDoubleArray
    new DFSModel(uid, weights)
  }

  override def transformSchema(schema: StructType): StructType = ???

  override def copy(extra: ParamMap): Estimator[DFSModel] = ???

  // -------- Utility Functions -----------------------------------

  private def QPFS(train: => RDD[Array[Byte]], nColumns:Int, nInstances:Int):Array[Double] = {

    val SU = DMI.SU(train, nColumns, nInstances, this.getMaxBin, this.getBatchSize)
    val nze = SU.zipWithIndex.filterNot(_._1.sum.isNaN).map(_._2)
    val nzeSU = SU.filterNot(_.sum.isNaN).map(a => a.filterNot(_ == 0))
    val ffSU = nzeSU.dropRight(1).map(row => row.dropRight(1))
    val fcSU = nzeSU.last.dropRight(1)

    val nzeWeights = QP(ffSU, fcSU).getWeights()
    val weights = Array.fill(nColumns - 1)(nzeWeights.min)
    for (i <- nze.dropRight(1).indices) weights(nze(i)) = nzeWeights(i)

    weights
  }
  private def SRFS(train: => RDD[Array[Byte]], nColumns:Int, nInstances:Int):Array[Double] = {

    val CMI = DMI.CMI(train, nColumns, nInstances, this.getMaxBin, this.getBatchSize)
    SR(CMI, isSymmetric = false).getWeights
  }
  private def TPFS(train: => RDD[Array[Byte]], nColumns:Int, nInstances:Int):Array[Double] = {
    val CMI = DMI.CMI(train, nColumns, nInstances, this.getMaxBin, this.getBatchSize)
    TP(CMI, nColumns - 1, isSymmetric = false).getWeights
  }
  private def transformColumnar(it: Iterator[Row], nColumn:Int): Iterator[Array[Byte]] = {

    val dataPartition = it.toVector
    val nc = nColumn
    val nr = dataPartition.length
    val DSc = Array.ofDim[Byte](nc, nr)

    dataPartition.zipWithIndex.foreach {
      case (vec, i) =>
        val row = vec.getAs[Any](this.getFeaturesCol) match {
          case v: DenseVector => v.toArray
          case v: SparseVector => v.toArray
        }
        for (j <- row.indices) DSc(j)(i) = row(j).toByte
        val idx = vec.fieldIndex(this.getLabelCol)
        DSc(nc - 1)(i) = vec.get(idx) match {
          case st: String => st.toByte
          case b: Byte => b
          case sh: Short => sh.toByte
          case d: Double => d.toByte
        }
    }

    DSc.toIterator
  }
}
