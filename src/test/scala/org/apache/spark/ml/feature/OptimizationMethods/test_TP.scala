package org.apache.spark.ml.feature.OptimizationMethods

import org.apache.spark.ml.feature.BasicTest
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class test_TP extends BasicTest {

  test("Traditional-QPFS") {

    val path = "src/test/scala/resources/data/Q.csv"
    val source = scala.io.Source.fromFile(path)
    val lines = source.getLines().toArray
    val Q = lines.map(s => s.split(",").map(_.toDouble))
    val nFeatures = Q.length
    val weights = TP(Q, nFeatures).getWeights
    val actualRank = weights.zipWithIndex.sortBy(_._1).reverse.map(_._2)
    val expectedRank = Array(10, 11, 9, 8, 14, 15, 5, 16, 12, 4, 6, 13, 7, 3, 17, 18, 2, 19, 0, 1, 20)

    assert(actualRank === expectedRank)
  }
}
