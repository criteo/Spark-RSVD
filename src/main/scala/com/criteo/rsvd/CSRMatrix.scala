package com.criteo.rsvd

import breeze.linalg.CSCMatrix

class CSRMatrix(val data: Array[Double],
                val rows: Int,
                val cols: Int,
                val rowPtrs: Array[Int],
                val colsIndices: Array[Int])
    extends Serializable {

  @transient lazy val csc =
    new CSCMatrix[Double](data, cols, rows, rowPtrs, colsIndices)

  def activeIterator: Iterator[((Int, Int), Double)] = {
    csc.activeIterator.map({ case ((i, j), v) => ((j, i), v) })
  }

  def t: CSRMatrix = {
    val builder = new CSRMatrix.Builder(cols, rows, initNnz = data.size)
    activeIterator.foreach({ case ((i, j), v) => builder.add(j, i, v) })
    builder.result
  }

  def apply(i: Int, j: Int): Double = {
    csc(j, i)
  }

}

object CSRMatrix {

  class Builder(rows: Int, cols: Int, initNnz: Int = 16) {
    private val _cscBuilder = new CSCMatrix.Builder[Double](cols, rows, initNnz)

    def add(row: Int, col: Int, value: Double): Unit = {
      _cscBuilder.add(col, row, value) // columns and rows are transposed !
    }

    def result: CSRMatrix = result(false, false)

    def result(keysAlreadyUnique: Boolean = false,
               keysAlreadySorted: Boolean = false): CSRMatrix = {
      val csc = _cscBuilder.result(keysAlreadyUnique, keysAlreadySorted)

      new CSRMatrix(csc.data, rows, cols, csc.colPtrs, csc.rowIndices)
    }
  }

}
