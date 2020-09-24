package org.apache.spark.ml.feature

import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

object DataSetSchema {

  trait baseSchema {

    def nColumn: Int

    def fIndexes: Range // feature indexes

    def cIndex: Integer // class index

    def name: String // name of data set

    def fNamePrefix: String = "col"

    def fNames: IndexedSeq[String] = for (i <- fIndexes) yield fNamePrefix + i

    def cName: String = "label"

    def Schema: StructType = {
      val nullable = true
      val structures = for (i <- fIndexes) yield StructField(fNamePrefix + i, DoubleType, nullable)
      StructType(structures :+ StructField("label", DoubleType, nullable))
    }
  }

  object Musk extends baseSchema {

    override def nColumn: Int = 167

    override def fIndexes: Range = 0 to 165

    override def cIndex: Integer = 166

    override def name: String = "Musk"
  }

  object Alpha extends baseSchema {

    override def nColumn: Int = 501

    override def fIndexes: Range = 0 to 499

    override def cIndex: Integer = 500

    override def name: String = "Alpha"

  }

}