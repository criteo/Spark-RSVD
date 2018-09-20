package com.criteo.rsvd

import breeze.linalg.{upperTriangular, DenseMatrix => BDM}
import org.scalatest.FunSuite

object SkinnyBlockMatrixTest {
  val expectedBlockIds: Set[Int] = (0 to 9).toSet
  val expectedMachinePrecisionWithDouble = 1e-15
  val numRows = 96
  val numCols = 7
  val blockSize = 10
  val rowBlocksPerPartition = 2
}

class SkinnyBlockMatrixTest extends FunSuite with PerTestSparkSession {
  import SkinnyBlockMatrixTest._

  test("toLocalMatrix should correctly return the concatenated matrix") {
    val numBlockRows = 10
    val numBlockCols = 3
    val block1 = new BDM(numBlockRows,
                         numBlockCols,
                         Array.fill(numBlockRows * numBlockCols)(1d))
    val block2 = new BDM(numBlockRows,
                         numBlockCols,
                         Array.fill(numBlockRows * numBlockCols)(2d))
    val block3 = new BDM(1, numBlockCols, Array.fill(numBlockCols)(3d))
    val concatMat = BDM.vertcat(block1, block2, block3)
    val expectedResult =
      new BDM[Double](concatMat.rows, concatMat.cols, concatMat.data)

    val blocks = sc.parallelize(
      Seq(
        (0, block1),
        (1, block2),
        (2, block3)
      ))

    val localMat =
      SkinnyBlockMatrix(blocks,
                        numBlockRows,
                        numBlockRows * 2 + 1,
                        numBlockCols,
                        2).toLocalMatrix

    assert(localMat === expectedResult)
  }

  test("randomMatrix should correctly initialize a new matrix") {
    val mat =
      SkinnyBlockMatrix.randomMatrix(numRows,
                                     numCols,
                                     blockSize,
                                     rowBlocksPerPartition,
                                     sc,
                                     seed = 158342769L)
    assert(mat.blocks.count() === Math.ceil(1.0 * numRows / blockSize))
    assert(
      mat.blocks.getNumPartitions ===
        Math.ceil(1.0 * numRows / blockSize / rowBlocksPerPartition))

    val blockIds =
      mat.blocks.map { case (rowId, _) => rowId }.toLocalIterator.toSet
    assert(blockIds === expectedBlockIds)

    val localMat = mat.toLocalMatrix
    assert(localMat.rows === numRows)
    assert(localMat.cols === numCols)
  }

  test("randomMatrix should produces the same matrix with the same seed") {
    val mat1 = SkinnyBlockMatrix.randomMatrix(numRows,
                                              numCols,
                                              blockSize,
                                              rowBlocksPerPartition,
                                              sc,
                                              seed = 1)

    val mat2 = SkinnyBlockMatrix.randomMatrix(numRows,
                                              numCols,
                                              blockSize,
                                              rowBlocksPerPartition,
                                              sc,
                                              seed = 1)

    assert(mat1.toLocalMatrix === mat2.toLocalMatrix)
  }

  test("randomMatrix should produces different matrices with different seeds") {
    val mat1 = SkinnyBlockMatrix.randomMatrix(numRows,
                                              numCols,
                                              blockSize,
                                              rowBlocksPerPartition,
                                              sc,
                                              seed = 1)

    val mat2 = SkinnyBlockMatrix.randomMatrix(numRows,
                                              numCols,
                                              blockSize,
                                              rowBlocksPerPartition,
                                              sc,
                                              seed = 2)

    assert(mat1.toLocalMatrix !== mat2.toLocalMatrix)
  }

  test(
    "singleBlockMultiply should correctly multiply and not change the partitioning") {
    val mat =
      SkinnyBlockMatrix.randomMatrix(numRows,
                                     numCols,
                                     blockSize,
                                     rowBlocksPerPartition,
                                     sc,
                                     seed = 158342769L)
    val (_, singleBlockMat) = mat.blocks.toLocalIterator.toSeq.head

    val expectedResult = mat.toLocalMatrix * singleBlockMat.t
    val result =
      mat.singleBlockMultiply(singleBlockMat.t, persistResult = false)
    val resultBreeze = result.toLocalMatrix

    assert(
      Utils.relativeNormError(expectedResult, resultBreeze) <=
        expectedMachinePrecisionWithDouble)

    assert(result.blocks.partitions.length === mat.blocks.partitions.length)
  }

  test("dot should work") {
    val matU =
      SkinnyBlockMatrix.randomMatrix(numRows,
                                     numCols,
                                     blockSize,
                                     rowBlocksPerPartition,
                                     sc,
                                     seed = 158342769L)
    val matV =
      SkinnyBlockMatrix.randomMatrix(numRows,
                                     numCols,
                                     blockSize,
                                     rowBlocksPerPartition,
                                     sc,
                                     seed = 158343494L)
    val localMat = matU.toLocalMatrix
    val localDot = matU.toLocalMatrix.t * matV.toLocalMatrix
    assert(
      Utils
        .relativeNormError(matU.dot(matV), localDot) < expectedMachinePrecisionWithDouble)
  }

  test(
    "Tall-and-skinny QR should produce Q with orthogonal columns " +
      "and the same number of partitions") {

    val mat =
      SkinnyBlockMatrix.randomMatrix(numRows,
                                     numCols,
                                     blockSize,
                                     rowBlocksPerPartition,
                                     sc,
                                     seed = 158342769L)
    val (q, _) = mat.qr
    assert(
      Utils.orthogonalizationError(q.toLocalMatrix) <=
        expectedMachinePrecisionWithDouble)
    assert(q.blocks.partitions.length === mat.blocks.partitions.length)
  }

  test(
    "Tall-and-skinny QR should do QR decomposition that restores the original matrix") {
    val mat =
      SkinnyBlockMatrix.randomMatrix(numRows,
                                     numCols,
                                     blockSize,
                                     rowBlocksPerPartition,
                                     sc,
                                     seed = 158342769L)
    val (q, r) = mat.qr

    val reconstructedMatrix = q.toLocalMatrix * r

    val error = Utils.relativeNormError(mat.toLocalMatrix, reconstructedMatrix)

    assert(r == upperTriangular(r))
    assert(error <= expectedMachinePrecisionWithDouble)
  }

  test("minus should work as expected") {
    val matX =
      SkinnyBlockMatrix.randomMatrix(numRows,
                                     numCols,
                                     blockSize,
                                     rowBlocksPerPartition,
                                     sc,
                                     seed = 158342769L)
    val matY =
      SkinnyBlockMatrix.randomMatrix(numRows,
                                     numCols,
                                     blockSize,
                                     rowBlocksPerPartition,
                                     sc,
                                     seed = 2344658L)

    val localMinus = matX.toLocalMatrix - matY.toLocalMatrix

    assert(matX.minus(matY).toLocalMatrix === localMinus)

  }
}
