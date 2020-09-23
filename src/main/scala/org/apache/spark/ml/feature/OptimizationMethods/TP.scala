package org.apache.spark.ml.feature.OptimizationMethods

import org.apache.spark.internal.Logging

/**
  * Truncated power method for sparse eigenvalue problem
  * Xiao-Tong Yuan, Tong Zhang, Truncated Power Method for Sparse Eigenvalue Problems, Technical Report, 2011
  * The implementation of code can be found in
  *   1) https://sites.google.com/site/xtyuan1980/publications. (original version)
  *   2) https://github.com/a736ii/EnhancingTPower
  */
class TP(A:Array[Array[Double]], cardinality:Int, maxIteration:Int = 50, optTol:Double = 1e-6, verbose:Boolean = false) extends Logging {

  def getWeights:Array[Double] = {

    logInfo("---------- Truncated Power Method ---------------")

    val idx = A.zipWithIndex.map { case (vec, d) => vec(d) }.zipWithIndex.sortBy(_._1).reverse.map(_._2)
    var x0 = Array.fill(A.length)(0.0)

    idx.take(cardinality).foreach(i => x0(i) = 1)
    val nx0 = norm(x0)
    x0 = x0.map(v => v / nx0)

    var x = x0
    // A * x0
    var s = A.map { row => row.zip(x).map(vv => vv._1 * vv._2).sum }
    // 2 * s
    var g = s.map(_ * 2)
    // x' * s
    var f = x.zip(s).map(vv => vv._1 * vv._2).sum

    x = truncate_operator(g, cardinality)
    var f_old = f
    var i = 1
    while (i < maxIteration) {

      // power step
      s = A.map { row => row.zip(x).map(vv => vv._1 * vv._2).sum }
      g = s.map(_ * 2)


      // truncate step
      x = truncate_operator(g, cardinality)
      f = x.zip(s).map(vv => vv._1 * vv._2).sum

      //% Output Log
      if (verbose)
        logInfo(s"Truncated power iteration: ($i , $f)")


      if (math.abs(f - f_old) < optTol) i = maxIteration

      f_old = f
      i = i + 1
    }

    x
  }

  private def norm(vec:Array[Double]):Double = math.sqrt(vec.map(v => v * v).sum)
  private def truncate_operator(values: Array[Double] , k:Int):Array[Double] = {

    val u = Array.fill[Double](values.length)(0.0)
    val idx = values.zipWithIndex.sortBy { case (v, _) => math.abs(v) }.map(_._2)
    val v_restrict = idx.take(k).map(i => values(i))
    val nv = norm(v_restrict)
    val nnv = v_restrict.map(_ / nv)
    (0 until k).foreach(i => u(idx(i)) = nnv(i))
    u
  }
}

object TP {
  def apply(Q: Array[Array[Double]], cardinality: Int): TP = new TP(Q, cardinality)
}
