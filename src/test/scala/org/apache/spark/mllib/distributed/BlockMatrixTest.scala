/*
 * Copyright (c) 2018 Criteo
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.distributed

import java.math.RoundingMode
import java.util.Random

import com.criteo.rsvd.PerTestSparkSession
import breeze.linalg.{diag, DenseMatrix => BDM, DenseVector => BDV}
import com.criteo.rsvd.{BlockMatrix, CSRMatrix, SkinnyBlockMatrix, Utils}
import com.google.common.math.IntMath
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{GridPartitioner, MatrixEntry}
import org.apache.spark.rdd.PartitionPruningRDDPartition
import org.scalatest.FunSuite

object BlockMatrixTest {
  val expectedMachinePrecisionWithDouble = 1e-15

  def createRandomMatrix(matHeight: Int,
                         matWidth: Int,
                         blockSize: Int,
                         partitionHeightInBlocks: Int,
                         partitionWidthInBlocks: Int,
                         sc: SparkContext,
                         seed: Long = 158342769L,
                         sparsity: Double = 0.25): BlockMatrix = {
    val totalSize = matHeight * matWidth
    val numNonZeroEntries = (totalSize * sparsity).toInt

    val entries = sc.parallelize(0 until numNonZeroEntries).map { idx =>
      val random = new Random(seed + idx)
      MatrixEntry(random.nextInt(matHeight),
                  random.nextInt(matWidth),
                  random.nextGaussian())
    }

    BlockMatrix.fromMatrixEntries(entries,
                                  matHeight = matHeight,
                                  matWidth = matWidth,
                                  blockSize,
                                  partitionHeightInBlocks,
                                  partitionWidthInBlocks)
  }

  def CSRMatrixZeros(rows: Int, cols: Int) =
    new CSRMatrix(new Array[Double](0),
                  rows,
                  cols,
                  new Array[Int](1),
                  new Array[Int](0))
}

class BlockMatrixTest extends FunSuite with PerTestSparkSession {
  import BlockMatrixTest._

  test("toLocalMatrix returns the concatenated matrix") {

    def buildSparseMatrixFromDense(dense: BDM[Double]): CSRMatrix = {
      val builder =
        new CSRMatrix.Builder(dense.rows, dense.cols, dense.rows * dense.cols)
      dense.foreachPair {
        case ((i, j), v) => builder.add(i, j, v)
      }
      builder.result
    }

    val numBlockDims = 10
    val lastBlockDims = 3

    val block00 = new BDM(numBlockDims,
                          numBlockDims,
                          Array.fill(numBlockDims * numBlockDims)(1d))

    val block01 = new BDM(numBlockDims,
                          lastBlockDims,
                          Array.fill(numBlockDims * lastBlockDims)(2d))

    val block10 = new BDM(lastBlockDims,
                          numBlockDims,
                          Array.fill(lastBlockDims * numBlockDims)(3d))

    val block11 =
      new BDM(lastBlockDims,
              lastBlockDims,
              Array.fill(lastBlockDims * lastBlockDims)(4d))

    val row0 = BDM.horzcat(block00, block01)
    val row1 = BDM.horzcat(block10, block11)
    val fullMat = BDM.vertcat(row0, row1)
    val expectedResult =
      new BDM[Double](fullMat.rows, fullMat.cols, fullMat.data)

    val numColPartitions = 2
    val rdd1 = sc.parallelize(Seq(
                                ((0, 0), buildSparseMatrixFromDense(block00)),
                                ((0, 1), buildSparseMatrixFromDense(block01))
                              ),
                              numColPartitions)

    val rdd2 = sc.parallelize(Seq(
                                ((1, 0), buildSparseMatrixFromDense(block10)),
                                ((1, 1), buildSparseMatrixFromDense(block11))
                              ),
                              numColPartitions)

    val localMat =
      BlockMatrix(
        Array(rdd1, rdd2),
        numBlockDims,
        matHeight = numBlockDims + lastBlockDims,
        matWidth = numBlockDims + lastBlockDims,
        partitionHeightInBlocks = 1,
        partitionWidthInBlocks = 1
      ).toLocalMatrix

    assert(localMat === expectedResult)
  }

  test("generate BlockMatrix generates a random matrix with given parameters") {

    val matSize = 95
    val blockSize = 10
    val partitionHeightInBlocks = 1
    val partitionWidthInBlocks = 1
    val mat = BlockMatrixTest.createRandomMatrix(matHeight = matSize,
                                                 matWidth = matSize,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc,
                                                 sparsity = 1.0)

    val localMat = mat.toLocalMatrix
    assert(localMat.rows === matSize && localMat.cols === matSize)
    assert(mat.blockSize === blockSize)
    assert(
      mat.blocksRDDs.length === IntMath
        .divide(matSize, blockSize, RoundingMode.CEILING))
    assert(mat.blocksRDDs.forall(rdd => rdd.getNumPartitions === rdd.count()))
  }

  test(
    "fromMatrixEntries produces a sparse square matrix with the given elements") {
    val matSize = 95
    val blockSize = 10
    val partitionHeightInBlocks = 1
    val partitionWidthInBlocks = 1

    val entries = Seq(
      (0, 0, 1), // goes to block (0,0) ie 0
      (blockSize - 1, 0, 1), // goes to block (0,0) ie 0
      (blockSize, 0, 2), // goes to block (1,0) ie 1
      (0, blockSize - 1, 1), // goes to block (0,0) ie 0
      (0, blockSize, 2), // goes to block (0,1) ie 10
      (blockSize - 1, blockSize - 1, 1), // goes to block (0,0) ie 0
      (blockSize, blockSize, 2), // goes to block (1,1) ie 11
      (matSize - 1, matSize - 1, 1), // goes to block (9,9) ie 99
      (matSize - 1, 0, 1), // goes to block (9,0) ie 9
      (0, matSize - 1, 1) // goes to block (0,9) ie 90
    )

    val numNonEmptyBlocksPerRow = Map(
      0 -> 3,
      1 -> 2,
      2 -> 0,
      3 -> 0,
      4 -> 0,
      5 -> 0,
      6 -> 0,
      7 -> 0,
      8 -> 0,
      9 -> 2
    )

    val matrixEntries = entries.map(e => MatrixEntry(e._1, e._2, e._3))
    val expectedMatrix =
      new BDM[Double](matSize, matSize, Array.fill(matSize * matSize)(0.0))
    entries.foreach(e => expectedMatrix.update(e._1, e._2, e._3))

    val mat = BlockMatrix.fromMatrixEntries(sc.parallelize(matrixEntries),
                                            matHeight = matSize,
                                            matWidth = matSize,
                                            blockSize,
                                            partitionHeightInBlocks,
                                            partitionWidthInBlocks)

    val localMat = mat.toLocalMatrix
    assert(localMat.rows === matSize && localMat.cols === matSize)
    assert(mat.blockSize === blockSize)
    assert(
      mat.blocksRDDs.length === IntMath
        .divide(matSize, blockSize, RoundingMode.CEILING))
    assert(mat.blocksRDDs.zipWithIndex.forall {
      case (rdd, idx) =>
        rdd.getNumPartitions === IntMath.divide(matSize,
                                                blockSize,
                                                RoundingMode.CEILING) &&
          rdd.count() === numNonEmptyBlocksPerRow(idx)
    })
    assert(localMat === expectedMatrix)
  }

  test("fromMatrixEntries produces a sparse square matrix with correct format") {
    val matSize = 7 // matrix size is 7*7
    val blockSize = 2 // a block is 2*2 items
    val partitionHeightInBlocks = 2 // a sub-rdd is every 2 rows of block
    val partitionWidthInBlocks = 3 // a partition is every 3 columns of blocks

    val nbRowOfRDDs = 2 // how many rows of sub-RDDs in the RDD ? ceil(7/(2*2)) = 2
    val nbColOfPartitions = 2 // how many columns of partitions in the RDD ? ceil(7/(2*3)) = 2

    val entries = Seq(
      // these will be in SubRDD0, partition0:
      (0, 0, 1), // goes to block (0,0) ie SubRDD0, partition0
      (1, 1, 9), // goes to block (0,0) ie SubRDD0, partition0
      (2, 0, 1), // goes to block (1,0) ie SubRDD0, partition0
      (0, 4, 1), // goes to block (0,2) ie SubRDD0, partition0
      // (this is creating 3 blocks in this partition)
      //
      (4, 0, 1), // goes to block (2,0) ie SubRDD1, partition0
      (6, 0, 1), // goes to block (3,0) ie SubRDD1, partition0
      //
      (0, 6, 1), // goes to block (0,3) ie SubRDD0, partition1
      //
      (6, 6, 1) // goes to block (3,3) ie SubRDD1, partition1
    )

    val blocksUsedPerPartitionRef = Map(
      //(RddId, PartitionId) -> how many blocks are in this partition
      (0, 0) -> 3,
      (1, 0) -> 2,
      (0, 1) -> 1,
      (1, 1) -> 1
    )

    val matrixEntries = entries.map(e => MatrixEntry(e._1, e._2, e._3))
    val expectedMatrix =
      new BDM[Double](matSize, matSize, Array.fill(matSize * matSize)(0.0))
    entries.foreach(e => expectedMatrix.update(e._1, e._2, e._3))

    val mat = BlockMatrix.fromMatrixEntries(sc.parallelize(matrixEntries),
                                            matHeight = matSize,
                                            matWidth = matSize,
                                            blockSize,
                                            partitionHeightInBlocks,
                                            partitionWidthInBlocks)

    val localMat = mat.toLocalMatrix
    assert(localMat === expectedMatrix)
    assert(localMat.rows === matSize && localMat.cols === matSize)
    assert(mat.blockSize === blockSize)
    assert(mat.blocksRDDs.length === nbRowOfRDDs)
    assert(mat.blocksRDDs.zipWithIndex.forall {
      case (rdd, _) =>
        rdd.getNumPartitions === nbColOfPartitions
    })

    val blocksUsedPerPartition = mat.blocksRDDs.zipWithIndex.flatMap {
      case (rdd, rddId) =>
        rdd
          .mapPartitionsWithIndex({
            case (partitionId, partition) =>
              Iterator((rddId, partitionId) -> partition.length)
          })
          .collect()
    }.toMap

    assert(blocksUsedPerPartitionRef === blocksUsedPerPartition)
  }

  test("Multiplying a sparse square matrix should produce correct results") {
    val matSize = 94
    val blockSize = 10
    val partitionHeightInBlocks = 2
    val partitionWidthInBlocks = 2
    val squareMat = BlockMatrixTest.createRandomMatrix(matHeight = matSize,
                                                       matWidth = matSize,
                                                       blockSize,
                                                       partitionHeightInBlocks,
                                                       partitionWidthInBlocks,
                                                       sc)

    val skinnyCols = 2
    val seed = 0
    val skinnyMat = SkinnyBlockMatrix.randomMatrix(matSize,
                                                   skinnyCols,
                                                   blockSize,
                                                   partitionWidthInBlocks,
                                                   sc,
                                                   seed)

    val result = squareMat
      .skinnyMultiply(skinnyMat, repartitionAndPersistSkinnyMatrix = false)
      .toLocalMatrix
    val expectedResult = squareMat.toLocalMatrix * skinnyMat.toLocalMatrix

    assert(
      Utils.relativeNormError(expectedResult.toDenseMatrix, result) <=
        expectedMachinePrecisionWithDouble)
  }

  test("Multiplying a sparse rectangular matrix should produce correct results") {
    val matHeight = 94
    val matWidth = 105
    val blockSize = 10
    val partitionHeightInBlocks = 2
    val partitionWidthInBlocks = 2
    val rectangularMat =
      BlockMatrixTest.createRandomMatrix(matHeight = matHeight,
                                         matWidth = matWidth,
                                         blockSize,
                                         partitionHeightInBlocks,
                                         partitionWidthInBlocks,
                                         sc)

    val skinnyCols = 2
    val seed = 0
    val skinnyMat = SkinnyBlockMatrix.randomMatrix(matWidth,
                                                   skinnyCols,
                                                   blockSize,
                                                   partitionWidthInBlocks,
                                                   sc,
                                                   seed)

    val result = rectangularMat
      .skinnyMultiply(skinnyMat, repartitionAndPersistSkinnyMatrix = false)
      .toLocalMatrix
    val expectedResult = rectangularMat.toLocalMatrix * skinnyMat.toLocalMatrix

    assert(
      Utils.relativeNormError(expectedResult.toDenseMatrix, result) <=
        expectedMachinePrecisionWithDouble)
  }

  test("transpose a square BlockMatrix") {
    val matSize = 94
    val blockSize = 10
    val partitionHeightInBlocks = 2
    val partitionWidthInBlocks = 2
    val mat = BlockMatrixTest.createRandomMatrix(matHeight = matSize,
                                                 matWidth = matSize,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc)

    val matTranspose = mat.transpose
    val expectedTranspose = mat.toLocalMatrix.t
    val expectedNumPartitions =
      IntMath.divide(matSize,
                     blockSize * partitionWidthInBlocks,
                     RoundingMode.CEILING)

    assert(matTranspose.toLocalMatrix === expectedTranspose)
    assert(matTranspose.blocksRDDs.length === mat.blocksRDDs.length)
    assert(matTranspose.blocksRDDs.forall(rdd =>
      rdd.getNumPartitions == expectedNumPartitions))
  }

  test("transpose a rectangular BlockMatrix") {
    val matHeight = 94
    val matWidth = 105
    val blockSize = 10
    val partitionHeightInBlocks = 2
    val partitionWidthInBlocks = 2
    val mat = BlockMatrixTest.createRandomMatrix(matHeight = matHeight,
                                                 matWidth = matWidth,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc)

    val matTranspose = mat.transpose
    val expectedTranspose = mat.toLocalMatrix.t
    val expectedNumPartitions =
      IntMath.divide(matHeight,
                     blockSize * partitionWidthInBlocks,
                     RoundingMode.CEILING)

    assert(matTranspose.toLocalMatrix === expectedTranspose)
    assert(matTranspose.blocksRDDs.length === 6)
    assert(mat.blocksRDDs.length === 5)
    assert(matTranspose.blocksRDDs.forall(rdd =>
      rdd.getNumPartitions == expectedNumPartitions))
  }

  test("splitIntoMultipleRDDs gives correct results") {

    val entries = for (i <- 0 until 10;
                       j <- 0 until 10)
      yield {
        (i, j) -> CSRMatrixZeros(5, 5)
      }

    val matSize = 50 //total input matrix is 50x50
    val blockSize = 5 //each block has 5 row / columns
    val numDimBlocks = matSize / blockSize //10 blocks are needed per dimension

    val partitionHeightInBlocks = 4 //split along y-axis into RDDs every 4 blocks, so it should split into 3
    val partitionWidthInBlocks = 2 //split along x-axis into partitions every 2 blocks, so it should split into 5

    val gridPartitioner =
      GridPartitioner(numDimBlocks,
                      numDimBlocks,
                      partitionHeightInBlocks,
                      partitionWidthInBlocks)

    val entriesRDD = sc.parallelize(entries).partitionBy(gridPartitioner)

    //check that we indeed have 15 partitions: 5 along x-axis and 3 along y-axis
    assert(entriesRDD.partitions.length == 15)

    val resultArray =
      BlockMatrix.splitIntoMultipleRDDs(entriesRDD,
                                        numDimBlocks,
                                        numDimBlocks,
                                        partitionHeightInBlocks)

    //grid partitioner split x-axis into 3, so we should have 3 sub-RDDs
    assert(resultArray.length == 3)

    val mappings = resultArray.map(_.partitions.map(p =>
      p.asInstanceOf[PartitionPruningRDDPartition] match {
        case p2 => (p2.index, p2.parentSplit.index)
    }))

    //check mapping between old partition ids and new partition ids
    assert(
      mappings ===
        Array(Array((0, 0), (1, 3), (2, 6), (3, 9), (4, 12)),
              Array((0, 1), (1, 4), (2, 7), (3, 10), (4, 13)),
              Array((0, 2), (1, 5), (2, 8), (3, 11), (4, 14)))
    )

  }

  test(
    "skinnyMultiplyMinusSVD should work for rectangular matrices, both direct and transpose") {
    val matHeight = 30
    val matWidth = 50
    val blockSize = 10
    val partitionHeightInBlocks = 2
    val partitionWidthInBlocks = 1
    val mat = BlockMatrixTest.createRandomMatrix(matHeight = matHeight,
                                                 matWidth = matWidth,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc)

    val matLocal = mat.toLocalMatrix

    val matTranspose = mat.transpose

    val embDim = 7
    val u =
      SkinnyBlockMatrix.randomMatrix(
        matHeight,
        embDim,
        blockSize,
        partitionWidthInBlocks,
        sc,
        123) // number of blocks per partition in a vector is always the same
    val uLocal = u.toLocalMatrix

    val v =
      SkinnyBlockMatrix.randomMatrix(matWidth,
                                     embDim,
                                     blockSize,
                                     partitionWidthInBlocks,
                                     sc,
                                     234)
    val vLocal = v.toLocalMatrix

    val sigma = BDV.rand[Double](embDim)

    val x =
      SkinnyBlockMatrix.randomMatrix(matWidth,
                                     embDim,
                                     blockSize,
                                     partitionWidthInBlocks,
                                     sc,
                                     345)
    val xLocal = x.toLocalMatrix

    val y =
      SkinnyBlockMatrix.randomMatrix(matHeight,
                                     embDim,
                                     blockSize,
                                     partitionWidthInBlocks,
                                     sc,
                                     456)
    val yLocal = y.toLocalMatrix

    assert(
      Utils.relativeNormError(
        matLocal * xLocal - uLocal * diag(sigma) * vLocal.t * xLocal,
        mat
          .skinnyMultiplyMinusSVD(x, u, sigma, v, false, false)
          .toLocalMatrix) < expectedMachinePrecisionWithDouble)

    assert(
      Utils.relativeNormError(
        matLocal.t * yLocal - vLocal * diag(sigma) * uLocal.t * yLocal,
        matTranspose
          .skinnyMultiplyMinusSVD(y, u, sigma, v, false, true)
          .toLocalMatrix) < expectedMachinePrecisionWithDouble)

  }
}
