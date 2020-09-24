package org.apache.spark.ml.feature.SGMI

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.rdd.RDD

/**
  *
  * Distributed Mutual Information contains three main functions MI, SU and CMI
  * MI = Mutual Information
  * SU = Symmetric Uncertainty
  * CMI = Conditional Mutual Information
  *
  * The main diameter of DCMI matrix is equal to the Mutual Information between variable X and class label is known as C
  * and other cells include the Conditional Mutual Information, among variables X, C, and Y
  * It is assume that the input matrix is in a transpose format such that each variable is placed in a row of the data matrix,
  * and that the class label is placed in the last row.
  * CMI(X;C | X) = MI(X,C),
  * CMI(X;C | Y) = H(X,Y) + H(Y,C) - H(X,Y,C)-H(Y)
  * MI(X,C) = H(X) + H(C) - H(X,C)
  * SU(X,Y) = 2 * MI(X,Y) / (H(X) + H(y))
  */

object DMI extends Serializable with Logging {

  def MI(data: => RDD[Array[Byte]], nColumns: Int, nInstances: Int, maxBin: Int, batchSize: Double): Array[Array[Double]] = {

    val start: Long = System.currentTimeMillis()
    val nBatch = math.round(1 / batchSize).toInt
    val totComputation = nColumns * (nColumns + 1) / 2
    val ranges = SplitRange(0 to totComputation, nBatch)
    val miMatrix = Array.ofDim[Double](nColumns, nColumns)

    val histogramXY = ranges.flatMap {
      range =>
        val xyIndexes = (for (x <- 0 until nColumns; y <- x until nColumns) yield (x, y))
          .zipWithIndex.filter { case (_, idx) => idx >= range.head && idx <= range.last }.map(_._1)

        data.mapPartitions {
          partition =>
            val rows = partition.toArray
            xyIndexes.map { case (x, y) => (x, y) -> Histogram(rows(x), rows(y), maxBin) }.toIterator
        }
          .reduceByKey { case (vec1, vec2) => AggregateSparseVectors(vec1.toSparse, vec2.toSparse) }
          .mapValues(v => Entropy(v.toArray, nInstances))
          .collect()
    }.toMap

    for (x <- 0 until nColumns; y <- x until nColumns) {
      if (x == y)
        miMatrix(x)(x) = histogramXY((x, x))
      else {
        //(Hx + Hy - Hxy)
        miMatrix(x)(y) = histogramXY((x, x)) + histogramXY((y, y)) - histogramXY((x, y))
        miMatrix(y)(x) = miMatrix(x)(y)
      }
    }

    logInfo(s"Distributed Mutual Information takes: ${System.currentTimeMillis() - start}")
    miMatrix
  }

  def CMI(data: => RDD[Array[Byte]], nColumns: Int, nInstances: Int, maxBin: Int, batchSize: Double): Array[Array[Double]] = {

    val nFeatures = nColumns - 1
    val z = nColumns - 1
    val nBatch = math.round(1 / batchSize).toInt
    val totComputation = nFeatures * (nFeatures - 1) / 2
    val ranges = SplitRange(0 to totComputation, nBatch)
    val cmiMatrix = Array.ofDim[Double](nFeatures, nFeatures)

    val start = System.currentTimeMillis()
    val xyzEntropy = ranges.flatMap {
      range =>

        val xyIndexes = (for (x <- 0 until nFeatures; y <- (x + 1) until nFeatures) yield (x, y))
          .zipWithIndex.filter { case (_, idx) => idx >= range.head && idx <= range.last }.map(_._1)

        data.mapPartitions {
          partition =>
            val rows = partition.toArray
            xyIndexes.map { case (x, y) => (x, y) -> Histogram(rows(x), rows(y), rows(z), maxBin) }.toIterator
        }
          .reduceByKey { case (vec1, vec2) => AggregateSparseVectors(vec1.toSparse, vec2.toSparse) }
          .flatMap {
            case ((x, y), frq) =>

              var res = List((x, y, z) -> Entropy(frq.toArray, nInstances))

              val frqCube = frq.toArray.grouped(maxBin).toArray.grouped(maxBin).toArray
              val fxy = frqCube.flatMap(m => m.map(v => v.sum))
              res = res :+ (x, y, y) -> Entropy(fxy, nInstances)

              if (y - x == 1) {

                val fx = frqCube.map(m => m.flatten.sum)
                val fxz = frqCube.flatMap(m => m.transpose.map(v => v.sum))
                res = res :+ (x, x, x) -> Entropy(fx, nInstances)
                res = res :+ (x, z, z) -> Entropy(fxz, nInstances)
              }

              if (y - x == 1 && nFeatures - y == 1) {

                val zvy = Array.ofDim[Double](maxBin)
                val fy = frqCube.map(m => m.map(_.sum)).foldLeft(zvy) { case (a1, a2) => a1.zip(a2).map { case (v1, v2) => v1 + v2 } }
                res = res :+ (y, y, y) -> Entropy(fy, nInstances)

                val zvz = Array.ofDim[Double](maxBin)
                val fz = frqCube.map(m => m.transpose.map(_.sum)).foldLeft(zvz) { case (a1, a2) => a1.zip(a2).map { case (v1, v2) => v1 + v2 } }
                res = res :+ (z, z, z) -> Entropy(fz, nInstances)

                val zvyz = Array.ofDim[Double](maxBin * maxBin)
                val fyz = frqCube.map(m => m.flatten).foldLeft(zvyz) { case (a1, a2) => a1.zip(a2).map { case (v1, v2) => v1 + v2 } }
                res = res :+ (y, z, z) -> Entropy(fyz, nInstances)
              }

              res

          }.collect()

    }.toMap

    for (x <- 0 until nFeatures; y <- x until nFeatures) {
      if (x == y)
        cmiMatrix(x)(x) = xyzEntropy((x, x, x)) + xyzEntropy((z, z, z)) - xyzEntropy((x, z, z)) //(Hx + Hz - Hxz)
      else {
        //(Hxy + Hyz - Hxyz - Hy)
        cmiMatrix(x)(y) = xyzEntropy((x, y, y)) + xyzEntropy((y, z, z)) - xyzEntropy((x, y, z)) - xyzEntropy((y, y, y))
        //(Hxy + Hxz - Hxyz - Hx)
        cmiMatrix(y)(x) = xyzEntropy((x, y, y)) + +xyzEntropy((x, z, z)) - xyzEntropy((x, y, z)) - xyzEntropy((x, x, x))
      }
    }

    logInfo(s"Distributed Conditional Mutual Information takes: ${System.currentTimeMillis() - start}")
    cmiMatrix
  }

  def SU(data: => RDD[Array[Byte]], nColumns: Int, nInstances: Int, maxBin: Int, batchSize: Double): Array[Array[Double]] = {

    val miMatrix = MI(data, nColumns, nInstances, maxBin, batchSize)
    val suMatrix = Array.ofDim[Double](nColumns, nColumns)
    for (i <- 0 until nColumns; j <- i until nColumns) {
      if (i == j)
        suMatrix(i)(i) = 1
      else {
        suMatrix(i)(j) = 2 * miMatrix(i)(j) / (miMatrix(i)(i) + miMatrix(j)(j))
        suMatrix(j)(i) = suMatrix(i)(j)
      }
    }
    suMatrix
  }

  private def Entropy(frequencies: Array[Double], nInstances: Double): Double = {
    frequencies.filter(_ > 0)
      .map(v => v / nInstances)
      .map(p => -1 * p * math.log(p))
      .sum / math.log(2)
  }

  private def Entropy(frequencies: linalg.Vector, nInstances: Double): Double = {

    frequencies.toSparse.values
      .map(v => v / nInstances)
      .map(p => -1 * p * math.log(p))
      .sum / math.log(2)
  }

  private def Histogram(vec1: => Array[Byte], vec2: => Array[Byte], maxBin: Int): linalg.Vector = {

    val frequencies = Array.ofDim[Double](maxBin, maxBin)
    (vec1, vec2).zipped.foreach {
      case (v1, v2) =>
        frequencies(v1)(v2) += 1
    }

    val ff = frequencies.flatten
    val (values, indices) = ff.zipWithIndex.filter(_._1 > 0).unzip
    new SparseVector(ff.length, indices, values).compressed
  }

  private def Histogram(vec1: => Array[Byte], vec2: => Array[Byte], vec3: => Array[Byte], maxBin: Int): linalg.Vector = {

    val frequencies = Array.ofDim[Double](maxBin, maxBin, maxBin)
    (vec1, vec2, vec3).zipped.foreach {
      case (v1, v2, v3) =>
        frequencies(v1)(v2)(v3) += 1
    }
    val ff = frequencies.flatten.flatten
    val (values, indices) = ff.zipWithIndex.filter(_._1 > 0).unzip
    new SparseVector(ff.length, indices, values).compressed
  }

  private def AggregateSparseVectors(vec1: => SparseVector, vec2: => SparseVector): linalg.Vector = {
    val indices = vec1.indices.toSet.union(vec2.indices.toSet)
    val values = indices.map(idx => idx -> (vec1(idx) + vec2(idx))).toSeq
    Vectors.sparse(vec1.size, values).compressed
  }

  private def SplitRange(r: Range, chunks: Int): Seq[Range] = {
    if (r.step != 1)
      throw new IllegalArgumentException("Range must have step size equal to 1")

    val nChunks = scala.math.max(chunks, 1)
    val chunkSize = scala.math.max(r.length / nChunks, 1)
    val starts = r.by(chunkSize).take(nChunks)
    val ends = starts.map(_ - 1).drop(1) :+ r.end
    starts.zip(ends).map(x => x._1 to x._2)
  }
}
