package com.criteo.rsvd

import org.apache.spark.Partitioner

/**
  * A grid partitioner, which uses a regular grid to partition coordinates.
  *
  * @param rows Number of rows.
  * @param cols Number of columns.
  * @param rowsPerPart Number of rows per partition, which may be less at the bottom edge.
  * @param colsPerPart Number of columns per partition, which may be less at the right edge.
  * This is a copy of the same class in package org.apache.spark.mllib.linalg.distributed which is private
  */
class GridPartitioner(val rows: Int,
                      val cols: Int,
                      val rowsPerPart: Int,
                      val colsPerPart: Int)
    extends Partitioner {

  require(rows > 0)
  require(cols > 0)
  require(rowsPerPart > 0)
  require(colsPerPart > 0)

  private val rowPartitions = math.ceil(rows * 1.0 / rowsPerPart).toInt
  private val colPartitions = math.ceil(cols * 1.0 / colsPerPart).toInt

  override val numPartitions: Int = rowPartitions * colPartitions

  /**
    * Returns the index of the partition the input coordinate belongs to.
    *
    * @param key The partition id i (calculated through this method for coordinate (i, j) in
    *            `simulateMultiply`, the coordinate (i, j) or a tuple (i, j, k), where k is
    *            the inner index used in multiplication. k is ignored in computing partitions.
    * @return The index of the partition, which the coordinate belongs to.
    */
  override def getPartition(key: Any): Int = {
    key match {
      case i: Int => i
      case t: Tuple2[Int, Int] => // To remove boxing (which happens with pattern matching)
        getPartitionId(t._1, t._2)
      case (i: Int, j: Int, _: Int) =>
        getPartitionId(i, j)
      case _ =>
        throw new IllegalArgumentException(s"Unrecognized key: $key.")
    }
  }

  /** Partitions sub-matrices as blocks with neighboring sub-matrices. */
  private def getPartitionId(i: Int, j: Int): Int = {
    require(0 <= i && i < rows, s"Row index $i out of range [0, $rows).")
    require(0 <= j && j < cols, s"Column index $j out of range [0, $cols).")
    i / rowsPerPart + j / colsPerPart * rowPartitions
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: GridPartitioner =>
        (this.rows == r.rows) && (this.cols == r.cols) &&
          (this.rowsPerPart == r.rowsPerPart) && (this.colsPerPart == r.colsPerPart)
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(rows: java.lang.Integer,
                                            cols: java.lang.Integer,
                                            rowsPerPart: java.lang.Integer,
                                            colsPerPart: java.lang.Integer)
  }
}

object GridPartitioner {

  /** Creates a new [[GridPartitioner]] instance. */
  def apply(rows: Int,
            cols: Int,
            rowsPerPart: Int,
            colsPerPart: Int): GridPartitioner = {
    new GridPartitioner(rows, cols, rowsPerPart, colsPerPart)
  }
}
