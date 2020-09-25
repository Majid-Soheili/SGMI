package ir

import org.apache.spark.sql.types.{ByteType, DoubleType, StructField, StructType}

sealed trait DataSetSchema extends Serializable {
  def nColumn: Int

  def fIndexes: Range // feature indexes

  def cIndex: Integer // class index

  def name: String // name of data set

  def fNamePrefix: String = "col"

  def fNames: IndexedSeq[String] = for (i <- fIndexes) yield fNamePrefix + i

  def cName: String = "label"

  def hasNegativeLabel: Boolean

  def numClasses: Int

  def categoricalFeaturesInfo: Seq[Int]

  def continuousFeaturesInfo: Seq[Int]

  def Schema: StructType = {
    val nullable = true
    val structures = for (i <- fIndexes) yield StructField(fNamePrefix + i, DoubleType, nullable)
    StructType(structures :+ StructField("label", DoubleType, nullable))
  }

  def isBinary: Boolean = false

  def binaryRecordLength: Int = 0

}
object DataSetSchema {

  case object Alpha extends DataSetSchema {

    override def nColumn: Int = 501

    override def fIndexes: Range = 0 to 499

    override def cIndex: Integer = 500

    override def name: String = "Alpha"

    override def hasNegativeLabel: Boolean = true

    override def numClasses: Int = 2

    override def categoricalFeaturesInfo: Seq[Int] = Seq.empty

    def continuousFeaturesInfo: Seq[Int] = fIndexes
  }

  case object Beta extends DataSetSchema {

    override def nColumn: Int = 501

    override def fIndexes: Range = 0 to 499

    override def cIndex: Integer = 500

    override def name: String = "Beta"

    override def hasNegativeLabel: Boolean = true

    override def numClasses: Int = 2

    override def categoricalFeaturesInfo: Seq[Int] = Seq.empty

    def continuousFeaturesInfo: Seq[Int] = fIndexes
  }

  case object Delta extends DataSetSchema {

    override def nColumn: Int = 501

    override def fIndexes: Range = 0 to 499

    override def cIndex: Integer = 500

    override def name: String = "Delta"

    override def hasNegativeLabel: Boolean = true

    override def numClasses: Int = 2

    override def categoricalFeaturesInfo: Seq[Int] = Seq.empty

    def continuousFeaturesInfo: Seq[Int] = fIndexes
  }

  object OCR extends DataSetSchema {

    override def nColumn: Int = 1157

    override def fIndexes: Range = 0 to 1155

    override def cIndex: Integer = 1156

    override def name: String = "OCR"

    override def hasNegativeLabel: Boolean = true

    override def numClasses: Int = 2

    override def categoricalFeaturesInfo: Seq[Int] = Seq.empty

    def continuousFeaturesInfo: Seq[Int] = fIndexes.toSeq

    override def Schema: StructType = {
      val nullable = true
      val structures = for (i <- fIndexes) yield StructField(fNamePrefix + i, ByteType, nullable)
      StructType(structures :+ StructField("label", ByteType, nullable))
    }

    override def binaryRecordLength: Int = 1157

    override def isBinary: Boolean = true
  }

  object FD extends DataSetSchema {

    override def nColumn: Int = 901

    override def fIndexes: Range = 0 to 899

    override def cIndex: Integer = 900

    override def name: String = "FD"

    override def hasNegativeLabel: Boolean = true

    override def numClasses: Int = 2

    override def Schema: StructType = {
      val nullable = false
      val structures = Seq.tabulate(nColumn - 1) { i => StructField(fNamePrefix + i, ByteType, nullable) }
      StructType(structures :+ StructField("label", ByteType, nullable))
    }

    override def categoricalFeaturesInfo: Seq[Int] = Seq.empty

    def continuousFeaturesInfo: Seq[Int] = fIndexes

    override def binaryRecordLength: Int = 901

    override def isBinary: Boolean = true
  }

  object Epsilon extends DataSetSchema {

    override def nColumn: Int = 2001

    override def fIndexes: Range = 0 to 1999

    override def cIndex: Integer = 2000

    override def name: String = "Epsilon"

    override def hasNegativeLabel: Boolean = true

    override def numClasses: Int = 2

    override def categoricalFeaturesInfo: Seq[Int] = Seq.empty

    def continuousFeaturesInfo: Seq[Int] = fIndexes
  }

  object ECBDL extends DataSetSchema {

    override def nColumn: Int = 632

    override def fIndexes: Range = 0 to 630

    override def cIndex: Integer = 631

    override def name: String = "ECBDL"

    override def hasNegativeLabel: Boolean = false

    override def numClasses: Int = 2

    override def categoricalFeaturesInfo: Seq[Int] = (3 to 20) ++ (39 to 92) ++ (131 to 150)

    override def continuousFeaturesInfo: Seq[Int] = fIndexes.diff(categoricalFeaturesInfo)
  }
}