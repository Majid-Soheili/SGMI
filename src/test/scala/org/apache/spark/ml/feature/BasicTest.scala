package org.apache.spark.ml.feature

import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.FunSuite
import org.scalatest.prop.Checkers

class BasicTest extends FunSuite with Checkers {

  implicit val doubleEquality: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(0.0001)

  implicit val matrixEq: Equality[Array[Array[Double]]] = new Equality[Array[Array[Double]]] {
    def areEqual(a: Array[Array[Double]], b: Any): Boolean =
      b match {
        case p: Array[Array[Double]] => {

          val x = a.flatten
          val y = p.flatten

          var i = 1
          var equal = true
          while (i < x.length && equal) {
            equal = x(i) === y(i)
            i = i + 1
          }
          equal
        }
        case _ => false
      }
  }

  implicit val arrayEq: Equality[Array[Double]] = new Equality[Array[Double]] {
    def areEqual(a: Array[Double], b: Any): Boolean =
      b match {
        case p: Array[Double] => {

          var i = 1
          var equal = true
          while (i < a.length && equal) {
            equal = a(i) === p(i)
            i = i + 1
          }
          equal
        }
        case _ => false
      }
  }

}
