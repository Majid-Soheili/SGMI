package org.apache.spark.ml.feature.OptimizationMethods

import org.apache.spark.ml.feature.BasicTest
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class test_QP extends  BasicTest {

  test("QP-H") {

    val path = "src/test/scala/resources/data/H.csv"
    val source = scala.io.Source.fromFile(path)
    val lines = source.getLines().toArray
    val H = lines.map(s => s.split(",").map(_.toDouble))

    val ffSU = H.dropRight(1).map(row => row.dropRight(1))
    val fcSU = H.last.dropRight(1)

    val weights = QP(ffSU, fcSU).getWeights()
    val actualRank = weights.zipWithIndex.sortBy(_._1).reverse.map(_._2)
    val expectedRank = Array(10, 9, 11, 8, 15, 16, 5, 14, 4, 12, 17, 13, 7, 3, 6, 18, 2, 19, 0, 20, 1)
    assert(actualRank === expectedRank)
  }
}
