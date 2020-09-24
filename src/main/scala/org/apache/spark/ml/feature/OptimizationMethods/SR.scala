package org.apache.spark.ml.feature.OptimizationMethods

import breeze.linalg.{eigSym, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.internal.Logging


/**
  * Spectral Relaxation
  */
class SR(H:Array[Array[Double]]) extends Logging {

  def getWeights: Array[Double] = {

    logInfo("---------- Spectral Relaxation Method ---------------")

    val start: Long = System.currentTimeMillis()
    val nColumns = H.length
    val A: BDM[Double] = BDM(H: _*)
    val es = eigSym(A)
    val lastFeature: Int = nColumns - 1
    val ev = es.eigenvectors(::, lastFeature) // biggest eigenvector is placed on the end of list
    val weights = NormalizeVector(ev).toArray
    logInfo(s"Spectral relaxation method takes ${System.currentTimeMillis() - start} ms")

    weights
  }

  private def NormalizeVector(vec: BDV[Double]): BDV[Double] = {
    val norm = vec.foldLeft(0.0)((b, v) => b + v * v)
    vec.mapValues(v => math.abs(v / norm))
  }

}

object SR {

  def apply(H: Array[Array[Double]], isSymmetric: Boolean = true, isNormalized: Boolean = false): SR = {
    if (isSymmetric && isNormalized)
      new SR(H)
    else {

      val sH = if (isSymmetric) H else SymmetricMatrix(H)
      val nsH = if (isNormalized) sH else NormalizeMatrix(sH)
      new SR(nsH)

    }
  }

  private def SymmetricMatrix(mat: => Array[Array[Double]]): Array[Array[Double]] = {
    val m_t = mat.transpose
    mat.zip(m_t).map { case (a1, a2) => a1.zip(a2).map { case (v1, v2) => (v1 + v2) / 2.0 } }
  }

  private def NormalizeMatrix(mat: => Array[Array[Double]]): Array[Array[Double]] = {

    val minValue = mat.map(a => a.min).min
    val maxValue = mat.map(a => a.max).max
    val diff = maxValue - minValue
    mat.map(a => a.map(v => (v - minValue) / diff))
  }

}
