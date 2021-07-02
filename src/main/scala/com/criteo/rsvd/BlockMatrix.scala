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

package com.criteo.rsvd

import java.math.RoundingMode

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.google.common.math.{IntMath, LongMath}
import com.google.common.primitives.Ints
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.rdd.{PartitionPruningRDD, RDD, UnionRDD}
import org.apache.spark.storage.StorageLevel

case class BlockEntry(rowNumberInBlock: Int,
                      colNumberInBlock: Int,
                      value: Double)

object BlockMatrix extends StrictLogging {

  /**
    * Split a matrix RDD into several submatrices PartitionPruningRDDs.
    * The split is done by splitting the matrix along the row axis by doing RDDs of maximum [partitionHeightInBlocks] blocks.
    * Because the initial RDD partitionned by blocks, we can simply use PartitionPruningRDD to create one-to-one
    * dependencies with the initial RDD's partition and avoid a shuffle.
    */
  def splitIntoMultipleRDDs(
      inputRDD: RDD[((Int, Int), CSRMatrix)],
      numDimBlocksHeight: Int,
      numDimBlocksWidth: Int,
      partitionHeightInBlocks: Int): Array[RDD[((Int, Int), CSRMatrix)]] = {
    val parentPartitioner = inputRDD.partitioner.get

    val numRDDs = IntMath.divide(numDimBlocksHeight,
                                 partitionHeightInBlocks,
                                 RoundingMode.CEILING)

    val partition = (0 until numRDDs).map { rddIndex =>
      (for {
        colBlockIdx <- 0 until numDimBlocksWidth
        rowBlockIdx <- rddIndex * partitionHeightInBlocks until Math.min(
          (rddIndex + 1) * partitionHeightInBlocks,
          numDimBlocksHeight)
      } yield parentPartitioner.getPartition((rowBlockIdx, colBlockIdx))).toSet
    }

    require(
      Utils.isAPartition(partition,
                         0,
                         inputRDD.partitioner.get.numPartitions - 1),
      s"Partition ${partition.mkString(";")} is not a partition of [0,${inputRDD.partitioner.get.numPartitions - 1}]"
    )

    /** Generate numRDDs RDDs which only keep the relevant partitions from inputRDD.
      * For this to work it is important that the partition ordering stays the same when pruning partitions,
      * which is the case with PartitionPruningRDD (ie partition 0 in the new RDD maps to the lowest kept partition id
      * in the initial RDD, partition 1 maps to the second partition, etc).
      */
    partition.map { partitionIndices =>
      new PartitionPruningRDD(inputRDD, partitionIndices.contains)
    }.toArray
  }

  /**
    * fromMatrixEntries creates a BlockMatrix from RDD of entries.
    * First, it groups them by blocks and transforms into SparseMatrices and
    * then splits into multiple RDDs by a specified number of row blocks.
    */
  def fromMatrixEntries(entries: RDD[MatrixEntry],
                        matHeight: Long,
                        matWidth: Long,
                        blockSize: Int,
                        partitionHeightInBlocks: Int,
                        partitionWidthInBlocks: Int,
                        computeInputSize: Boolean = false): BlockMatrix = {

    require(blockSize > 0,
            s"blockSize needs to be greater than 0. blockSize: $blockSize")
    require(partitionHeightInBlocks > 0,
            s"partitionHeightInBlocks needs to be greater than 0. " +
              s"partitionHeightInBlocks: $partitionHeightInBlocks")
    require(partitionWidthInBlocks > 0,
            s"partitionWidthInBlocks needs to be greater than 0. " +
              s"partitionWidthInBlocks: $partitionWidthInBlocks")

    val numDimBlocksHeight =
      Ints.checkedCast(
        LongMath.divide(matHeight, blockSize, RoundingMode.CEILING))

    val numDimBlocksWidth =
      Ints.checkedCast(
        LongMath.divide(matWidth, blockSize, RoundingMode.CEILING))

    val gridPartitioner =
      GridPartitioner(numDimBlocksHeight,
                      numDimBlocksWidth,
                      partitionHeightInBlocks,
                      partitionWidthInBlocks)

    // TODO: Re-implement so that we don't need to re-create all entries (heavy GC)
    val blocks: RDD[((Int, Int), CSRMatrix)] = entries
      .map { entry =>
        if (entry.i > matHeight || entry.j > matWidth) {
          throw new SparkException(
            s"Matrix entry (${entry.i},${entry.j}) exceeds allowed matrix size of $matHeight * $matWidth ")
        }

        val blockRowIndex = (entry.i / blockSize).toInt
        val blockColIndex = (entry.j / blockSize).toInt

        val blockRowId = (entry.i % blockSize).toInt
        val blockColId = (entry.j % blockSize).toInt

        ((blockRowIndex, blockColIndex),
         BlockEntry(blockRowId, blockColId, entry.value))
      }
      .groupByKey(gridPartitioner)
      .mapPartitions(
        iter =>
          iter.map {
            case ((blockRowIndex, blockColIndex), entry) =>
              val effRows =
                math
                  .min(matHeight - blockRowIndex.toLong * blockSize, blockSize)
                  .toInt
              val effCols =
                math
                  .min(matWidth - blockColIndex.toLong * blockSize, blockSize)
                  .toInt

              val builder = new CSRMatrix.Builder(effRows, effCols)
              entry.foreach {
                case BlockEntry(i, j, v) => builder.add(i, j, v)
              }

              ((blockRowIndex, blockColIndex), builder.result)
        },
        preservesPartitioning = true
      )

    if (computeInputSize) {
      //an extra action is required to do this
      val inputSize = blocks
        .map {
          case (_, mat) => mat.data.length
        }
        .sum()
        .toLong
      logger.info(
        s"Initial input matrix has $inputSize elements, which makes it ${inputSize.toDouble / (matWidth * matHeight)} dense")
    }

    /** We split a RDD partitioned like so:
      * +---+---+---+
      * | 0 | 2 | 4 |
      * +---+---+---+
      * | 1 | 3 | 5 |
      * +---+---+---+
      *
      * into several (here 2) RDDs partitioned like so:
      * +---+---+---+
      * | 0 | 1 | 2 |
      * +---+---+---+
      *
      * The key point is that we are splitting the RDD by rows along *all* partition axis.
      * This ensures 2 properties:
      *  - before and after the split contains the same elements, so no shuffle is required
      *  - each "new" RDD only contains partitions along the column axis, ie contains only one line
      *    of partitions
      */
    val muxBlocks = splitIntoMultipleRDDs(blocks,
                                          numDimBlocksHeight =
                                            numDimBlocksHeight,
                                          numDimBlocksWidth = numDimBlocksWidth,
                                          partitionHeightInBlocks)

    BlockMatrix(
      muxBlocks,
      blockSize,
      matHeight = matHeight,
      matWidth = matWidth,
      partitionHeightInBlocks = partitionHeightInBlocks,
      partitionWidthInBlocks = partitionWidthInBlocks
    )
  }

  /**
    * localReduceMultiply does in-place multiplication for single zipped partition
    * of a BlockMatrix and SkinnyBlockMatrix.
    * @param leftBlocks - block-matrices from a BlockMatrix
    * @param rightBlocks - corresponding block-matrices from a SkinnyBlockMatrix
    * @param numOutputCols - number of columns in a SkinnyBlockMatrix
    * @return - result of the multiplication as blocks of new SkinnyBlockMatrix
    */
  def localReduceMultiply(leftBlocks: Iterator[((Int, Int), CSRMatrix)],
                          rightBlocks: Iterator[(Int, BDM[Double])],
                          numOutputCols: Int): Iterator[(Int, BDM[Double])] = {

    logger.info("Entering localReduceMultiply...")

    val leftBlocksMap = leftBlocks.toArray.toMap
    val rowIndices = leftBlocksMap.keySet.map(_._1).toArray

    val accumulators = rowIndices.map { rowIndex =>
      val currentRowBlockKey =
        leftBlocksMap.keySet.filter(_._1 == rowIndex).head
      rowIndex -> BDM.zeros[Double](leftBlocksMap(currentRowBlockKey).rows,
                                    numOutputCols)
    }.toMap

    var nRightBlocks = 0

    rightBlocks.foreach {
      case (colIndex, rightBlock) =>
        nRightBlocks += 1
        rowIndices.foreach { rowIndex =>
          leftBlocksMap.get((rowIndex, colIndex)) match {
            case Some(leftBlock) =>
              Utils.gemm(alpha = 1.0,
                         leftBlock,
                         rightBlock,
                         beta = 1.0,
                         accumulators(rowIndex))
            case None => None
          }
        }
    }

    logger.info(
      s"Finished localReduceMultiply! Multiplied ${leftBlocksMap.size} left blocks with $nRightBlocks right blocks to produce ${accumulators.size} result blocks.")

    accumulators.toIterator
  }
}

/**
  * BlockMatrix is an implementation of a block-matrix of a rectangular shape.
  * This class can be used to implement efficient rectangular matrix times
  * tall-and-skinny matrix multiplication.
  *
  * The data structure used is a multiplexed rdds of block matrices:
  *            |* * * * * *|
  *  Let A =   |* * * * * *|
  *            |* * * * * *|
  *            |* * * * * *|
  *            |* * * * * *|
  *            |* * * * * *|
  *
  * Then the spark-based representation will be:
  *     RDD1
  *            |* *|* *|* *|
  *            |* *|* *|* *|
  *     RDD2
  *            |* *|* *|* *|
  *            |* *|* *|* *|
  *     RDD3
  *            |* *|* *|* *|
  *            |* *|* *|* *|
  *
  *  The goal of this data structure is to have small enough RDDs, so that
  *  when we multiply it by a block from the SkinnyBlockMatrix, the dense
  *  end-result would still fit in memory. And column-partitioned RDD can be
  *  efficiently zipped (by zipPartitions) with the SkinnyBlockMatrix.
  */
case class BlockMatrix(blocksRDDs: Array[RDD[((Int, Int), CSRMatrix)]],
                       blockSize: Int,
                       matHeight: Long,
                       matWidth: Long,
                       partitionHeightInBlocks: Int,
                       partitionWidthInBlocks: Int) {

  import BlockMatrix._

  val numDimBlocksHeight: Int =
    Ints.checkedCast(
      LongMath.divide(matHeight, blockSize, RoundingMode.CEILING))

  val numDimBlocksWidth: Int =
    Ints.checkedCast(LongMath.divide(matWidth, blockSize, RoundingMode.CEILING))

  /**
    * skinnyMultiply does efficient BlockMatrix times SkinnyBlockMatrix
    * multiplication. This is the single heaviest operation in the R-SVD Job.
    * It does the no-shuffle join of every RDD in SquareBlockMatrix with
    * SkinnyBlockMatrix and does local multiplication.
    * The size of shuffle during the reduce stage is O(n**2 / d)
    * Where n is number of columns in A and d is the width of column partitions.
    * So in order to get feasible shuffle size we need to take d as large as possible.
    *
    * On the other hand, the memory consumption on the executors during the
    * localReduceMultiply is O(n*k/f + d)
    * Where k is number of columns in B and f is the number of RDDs in A.
    * Increasing f incurs overhead due to larger number of tasks.
    *
    * So by using more memory (higher d), we can reduce the shuffle size.
    * By taking larger f, we can decrease the memory consumption by sacrificing
    * some performance.
    *
    */
  def skinnyMultiply(
      skinnyMat: SkinnyBlockMatrix,
      repartitionAndPersistSkinnyMatrix: Boolean): SkinnyBlockMatrix = {
    require(
      matWidth == skinnyMat.numRows,
      "The number of columns of A and the number of rows " +
        s"of B must be equal. A.numCols: $matWidth, B.numRows: ${skinnyMat.numRows}. If you " +
        "think they should be equal, try setting the dimensions of A and B explicitly while " +
        "initializing them."
    )

    require(
      blockSize == skinnyMat.blockSize,
      s"The number of columns per block in A should" +
        s"be the same as the number of rows per block in B. A.blockSize: $blockSize and " +
        s"B.blockSize: ${skinnyMat.blockSize}"
    )

    val shuffledInMemoryRightBlocks = if (repartitionAndPersistSkinnyMatrix) {
      //partition scheme has to be compatible between matrix and skinny-matrix, otherwise, repartition
      //NB repartition won't happen if the RDD is already correctly partitionned
      skinnyMat.blocks
        .partitionBy(
          SingleDimensionPartitioner(numDimBlocksWidth, partitionWidthInBlocks))
        .persist(StorageLevel.MEMORY_AND_DISK_SER) // Persisting as serialized data as it is faster for shuffle io
    } else {
      skinnyMat.blocks
    }

    shuffledInMemoryRightBlocks
      .count() // Force the shuffle and memory persistence

    val multiplicationResult = blocksRDDs.map { leftBlocks =>
      leftBlocks
        .zipPartitions(shuffledInMemoryRightBlocks)({
          case (left, right) =>
            localReduceMultiply(left, right, skinnyMat.numCols)
        })
        .reduceByKey({ (a, b) =>
          a :+= b
        }, IntMath.divide(numDimBlocksHeight, 2, RoundingMode.CEILING))
    }.toSeq

    val newBlocks = skinnyMat.blocks.context
      .union(multiplicationResult)
      .persist(StorageLevel.DISK_ONLY)
    newBlocks.count() // Force the union

    SkinnyBlockMatrix(newBlocks,
                      blockSize,
                      matHeight,
                      skinnyMat.numCols,
                      partitionHeightInBlocks)
  }

  def skinnyMultiplyMinusSVD(x: SkinnyBlockMatrix,
                             u: SkinnyBlockMatrix,
                             sigma: BDV[Double],
                             v: SkinnyBlockMatrix,
                             repartitionAndPersistSkinnyMatrix: Boolean,
                             isTransposed: Boolean): SkinnyBlockMatrix = {
    val heightPartitioner =
      SingleDimensionPartitioner(numDimBlocksHeight, partitionWidthInBlocks)

    val multiplied =
      skinnyMultiply(x, repartitionAndPersistSkinnyMatrix).repartitionBy(
        heightPartitioner)

    val SVDCorrection =
      ReconstructionError.timesSVD(u, sigma, v, x, isTransposed)

    multiplied.minus(SVDCorrection)
  }

  /**
    * Transfer this BlockMatrix into a dense breeze matrix (on the driver)
    * @return the same matrix but available on the driver
    */
  def toLocalMatrix: BDM[Double] = {
    require(
      matHeight * matWidth < Int.MaxValue,
      "The length of the values array must be " +
        s"less than Int.MaxValue. Currently numRows * numCols: ${matHeight * matWidth}"
    )

    val localBlocks = blocksRDDs.flatMap(_.collect())
    val values = new Array[Double](Ints.checkedCast(matHeight * matWidth))
    localBlocks.foreach {
      case ((blockRowIndex, blockColIndex), submat) =>
        val rowOffset = blockRowIndex * blockSize
        val colOffset = blockColIndex * blockSize
        submat.activeIterator.foreach {
          case ((i, j), v) =>
            val indexOffset =
              Ints.checkedCast((j + colOffset) * matHeight + rowOffset + i)
            values(indexOffset) = v
        }
    }
    new BDM[Double](Ints.checkedCast(matHeight),
                    Ints.checkedCast(matWidth),
                    values)
  }

  /**
    * Transpose the BlockMatrix into an other BlockMatrix.
    * Note that the partition scheme is the same as the initial one,
    * so it won't be transposed. This means that the resulting matrix
    * will still have the same partitionWidthInBlocks and partitionHeightInBlocks
    * as the initial matrix.
    * @return the transposed matrix
    */
  def transpose: BlockMatrix = {

    val transposedBlocks = blocksRDDs.map(_.map {
      case ((blockRowIndex, blockColIndex), mat) =>
        ((blockColIndex, blockRowIndex), mat.t)
    })

    val transposeNumDimBlocksHeight = numDimBlocksWidth
    val transposeNumDimBlocksWidth = numDimBlocksHeight
    val transposeMatHeight = matWidth
    val transposeMatWidth = matHeight

    val gridPartitioner =
      GridPartitioner(transposeNumDimBlocksHeight,
                      transposeNumDimBlocksWidth,
                      partitionHeightInBlocks,
                      partitionWidthInBlocks)

    // we *don't* use sc.union() or UnionAwareRDD but UnionRDD to ensure that the previous partitions are not merged
    // (as the resulting partition split is much more shuffle-friendly)
    // TODO if partitionHeightInBlocks == partitionWidthInBlocks then no shuffle is needed as we are already
    // correctly partitionned
    val newBlocks =
      new UnionRDD(blocksRDDs.head.context, transposedBlocks)
        .partitionBy(gridPartitioner)

    val muxBlocks = splitIntoMultipleRDDs(
      newBlocks,
      numDimBlocksHeight = transposeNumDimBlocksHeight,
      numDimBlocksWidth = transposeNumDimBlocksWidth,
      partitionHeightInBlocks)

    BlockMatrix(
      muxBlocks,
      blockSize,
      //swap this
      matHeight = transposeMatHeight,
      matWidth = transposeMatWidth,
      //not this! partition scheme is not transposed
      partitionHeightInBlocks = partitionHeightInBlocks,
      partitionWidthInBlocks = partitionWidthInBlocks
    )
  }

  /**
    * Persist the blocks of this matrix to a given storage level
    * @param storageLevel is the storage level at which the block will be persisted.
    * @return this object
    */
  def persist(storageLevel: StorageLevel): this.type = {
    blocksRDDs.foreach(_.persist(storageLevel))
    this
  }
}
