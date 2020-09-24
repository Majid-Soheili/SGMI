package org.apache.spark.ml.feature.Model

import org.apache.spark.ml.Model
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType


/**
  * Distributed Feature Selection Model (DFSModel)
  */
class DFSModel private [ml](override val uid: String, private val featuresWeight: Array[Double])
  extends Model[DFSModel] with HasFeaturesCol with HasOutputCol with HasLabelCol {

  final val selectionThreshold = new DoubleParam(this, "selectionThreshold",
    "Represents a proportion of features to select, it should be bigger than 0.0 and less than or equal to 1.0. It is by default set to 0.10.",
    ParamValidators.inRange(0.0, 1.0, lowerInclusive = false, upperInclusive = true))
  setDefault(selectionThreshold -> 0.10)

  def getSelectionThreshold: Double = $(selectionThreshold)

  /** @group setParam */
  def setSelectionThreshold(value: Double): this.type = set(selectionThreshold, value)

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  def getRank: Array[Int] = {

    val weights = featuresWeight
    weights.zipWithIndex.sortBy(_._1).reverse.map(_._2)
  }

  def getSelectedFeatures: Array[Int] = {

    val sortedFeats = this.getRank
    // Slice according threshold
    sortedFeats.slice(0, (sortedFeats.length * $(selectionThreshold)).round.toInt)
  }

  override def transform(data: Dataset[_]): DataFrame = {
    val selectedFeatures: Array[Int] = this.getSelectedFeatures
    val slicer = new VectorSlicer()
      .setInputCol(featuresCol.name)
      .setOutputCol(outputCol.name)
      .setIndices(selectedFeatures)
    slicer.transform(data)
  }

  override def copy(extra: ParamMap): DFSModel = ???

  override def transformSchema(schema: StructType): StructType = ???
}


