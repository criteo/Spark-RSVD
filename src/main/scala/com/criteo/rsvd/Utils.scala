package com.criteo.rsvd

import breeze.linalg.{norm, DenseMatrix => BDM, DenseVector => BDV}
import com.github.fommil.netlib.BLAS.{getInstance => blas}

object Utils {

  def relativeNormError(matTarget: BDM[Double],
                        matApproximate: BDM[Double]): Double = {
    norm(matApproximate - matTarget) / norm(matTarget)
  }

  def relativeNormError(matTarget: BDV[Double],
                        matApproximate: BDV[Double]): Double = {
    norm(matApproximate - matTarget) / norm(matTarget)
  }

  def orthogonalizationError(Q: BDM[Double]): Double = {
    val eye = BDM.eye[Double](Q.cols)
    val diffs = Q.t * Q - eye
    norm(diffs) / norm(eye)
  }

  /**
    * The result of SVD is unique down to a sign-flip. In order to get deterministic
    * results, we can flip the signs when needed.
    */
  def deterministicSignsInplace(u: BDM[Double], v: BDM[Double]): Unit = {
    assert(u.cols == v.cols)

    (0 until u.cols).foreach { idx =>
      val vec = u(::, idx)
      if (vec(0) < 0) {
        u(::, idx) := -vec
        val vecV = v(::, idx)
        v(::, idx) := -vecV
      }
    }
  }

  implicit def canNorm_Double: norm.Impl2[BDM[Double], Double, Double] = {
    new norm.Impl2[BDM[Double], Double, Double] {
      def apply(v: BDM[Double], p: Double): Double = {
        if (p == 2) {
          var sq = 0.0
          v.foreachValue(x => sq += x * x)
          math.sqrt(sq)
        } else if (p == 1) {
          var sum = 0.0
          v.foreachValue(x => sum += math.abs(x))
          sum
        } else if (p == Double.PositiveInfinity) {
          var max = 0.0
          v.foreachValue(x => max = math.max(max, math.abs(x)))
          max
        } else if (p == 0) {
          var nnz = 0
          v.foreachValue(x => if (x != 0) nnz += 1)
          nnz
        } else {
          var sum = 0.0
          v.foreachValue(x => sum += math.pow(math.abs(x), p))
          math.pow(sum, 1.0 / p)
        }
      }
    }
  }

  def rowIter(dm: BDM[Double]): Iterator[BDV[Double]] = {
    require(!dm.isTranspose)
    Iterator.tabulate(dm.rows) { j =>
      val row = new Array[Double](dm.cols)
      blas.dcopy(dm.cols, dm.data, j, dm.rows, row, 0, 1)
      new BDV[Double](row)
    }
  }

  /**
    * Checks that a list of Set indeed forms a partition of the range [min, max]
    */
  private[rsvd] def isAPartition(partition: Seq[Set[Int]],
                                 min: Int,
                                 max: Int): Boolean = {
    var alreadyUsedItems = Set.empty[Int]

    var isPartition = true

    //check that each set is disjoint from the rest and non-empty
    partition.foreach { set =>
      isPartition = isPartition && alreadyUsedItems.intersect(set).isEmpty
      isPartition = isPartition && set.nonEmpty
      alreadyUsedItems = alreadyUsedItems.union(set)
    }

    //check that all items appear at least once
    isPartition = isPartition && (alreadyUsedItems.toList.sorted == (min to max).toList)

    isPartition
  }

  /**
    * C := alpha * A * B + beta * C
    * @param alpha a scalar to scale the multiplication A * B.
    * @param A the matrix A that will be left multiplied to B. Size of m x k. A is a CSR Matrix.
    * @param B the matrix B that will be left multiplied by A. Size of k x n. B.isTransposed must be false.
    * @param beta a scalar that can be used to scale matrix C.
    * @param C the resulting matrix C. Size of m x n. C.isTranspose must be false.
    * Implementation is a copy of org.apache.spark.mllib.linalg.gemm method for the branch where A is transposed
    */
  def gemm(alpha: Double,
           A: CSRMatrix,
           B: BDM[Double],
           beta: Double,
           C: BDM[Double]): Unit = {
    val mA: Int = A.rows
    val nB: Int = B.cols
    val kA: Int = A.cols
    val kB: Int = B.rows

    require(kA == kB,
            s"The columns of A don't match the rows of B. A: $kA, B: $kB")
    require(mA == C.rows,
            s"The rows of C don't match the rows of A. C: ${C.rows}, A: $mA")
    require(
      nB == C.cols,
      s"The columns of C don't match the columns of B. C: ${C.cols}, A: $nB")

    require(!B.isTranspose)
    require(!C.isTranspose)

    val Avals = A.data
    val Bvals = B.data
    val Cvals = C.data

    // Slicing is easy in this case. This is the optimal multiplication setting for sparse matrices
    var colCounterForB = 0
    while (colCounterForB < nB) {
      var rowCounterForA = 0
      val Cstart = colCounterForB * mA
      val Bstart = colCounterForB * kA
      while (rowCounterForA < mA) {
        var i = A.rowPtrs(rowCounterForA)
        val indEnd = A.rowPtrs(rowCounterForA + 1)
        var sum = 0.0
        while (i < indEnd) {
          sum += Avals(i) * Bvals(Bstart + A.colsIndices(i))
          i += 1
        }
        val Cindex = Cstart + rowCounterForA
        Cvals(Cindex) = beta * Cvals(Cindex) + sum * alpha
        rowCounterForA += 1
      }
      colCounterForB += 1
    }

  }

}
