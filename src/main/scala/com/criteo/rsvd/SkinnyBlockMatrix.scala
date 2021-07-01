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

import breeze.linalg.{inv, DenseMatrix => BDM, DenseVector => BDV}
import com.google.common.math.LongMath
import com.google.common.primitives.Ints
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import spire.syntax.cfor.cforRange

import scala.util.Random

object SkinnyBlockMatrix extends StrictLogging {

  /**
    * randomMatrix method generates a SkinnyBlockMatrix where each element is
    * drawn from an independent standard gaussian distribution. The choice of
    * distribution and it's parameters are arbitrary and should have almost
    * no effect on the result.
    * Here the goal is to generate a matrix with linearly independent columns
    * that will not be orthogonal to the top singular vectors of the matrix A.
    * So the choice of values is not important, only that the direction of
    * column-vectors are "random".
    *
    * @note This implementation can be improved (will require less power iterations)
    *       by initializing the columns with something that is better than a random guess,
    *       for instance, the previously found left-singular vectors would
    *       be a great start.
    *       The problem here would be that we can get "stuck" in a local minimum
    *       if other singular vectors will gain larger singular values and we
    *       will be orthogonal to them.
    *       Maybe just restart from random initialization from time to time?
    *       Anyway, it's only important if we do a lot of power iterations.
    */
  def randomMatrix(numRows: Long,
                   numCols: Int,
                   blockSize: Int,
                   numBlocksPerPartition: Int,
                   sc: SparkContext,
                   seed: Long): SkinnyBlockMatrix = {

    val numBlocks =
      Ints.checkedCast(
        LongMath.divide(numRows, blockSize, RoundingMode.CEILING))
    val lastRowBlockSize = LongMath.mod(numRows, blockSize)

    val rowBlocksPartitioner =
      SingleDimensionPartitioner(numBlocks, numBlocksPerPartition)

    val data = rowBlocksPartitioner
      .createCompatibleIndicesRDD(sc)
      .mapValues { idx =>
        val blockRows =
          if (idx == numBlocks - 1 && lastRowBlockSize != 0) lastRowBlockSize
          else blockSize

        val random = new Random(seed + idx.toLong)

        val data = new Array[Double](blockRows * numCols)
        cforRange(data.indices) { i =>
          data(i) = random.nextGaussian()
        }

        new BDM[Double](blockRows, numCols, data)
      }

    logger.info(
      s"Will generate a random matrix of size $numRows * $numCols, split into $numBlocks blocks.")
    logger.info(
      s"Will generate $numBlocks blocks of size $blockSize * $numCols, which is ${blockSize * numCols * 8.0 / (1024 * 1024)} MB each")

    SkinnyBlockMatrix(data,
                      blockSize,
                      numRows,
                      numCols,
                      partitionHeightInBlocks = numBlocksPerPartition)
  }

}

/**
  * SkinnyBlockMatrix is an implementation of a block-matrix of a specific
  * (Tall-And-Skinny) shape, meaning that the number of rows >> number of columns.
  * This class can be used to implement specific algorithms:
  * SkinnyBlockMatrix times a single block multiplication and
  * Indirect tall-and-skinny QR algorithm
  * (http://inside.mines.edu/~pconstan/docs/constantine-mrtsqr.pdf)
  * This matrix is always dense and is never split along the column dimensions.
  */
case class SkinnyBlockMatrix(blocks: RDD[(Int, BDM[Double])],
                             blockSize: Int,
                             numRows: Long,
                             numCols: Int,
                             partitionHeightInBlocks: Int) {

  require(
    numCols <= blockSize,
    s"SkinnyBlockMatrix has to have numCols <= blockSize. Currently: numCols=$numCols, blockSize=$blockSize")

  /**
    * Access to the SingleDimensionPartitioner of this matrix when it's how the RDD is partitioned.
    * It's not always the case as the call to BlockMatrix.skinnyMultiply will remove the partitioning
    */
  def partitioner: Option[SingleDimensionPartitioner] =
    blocks.partitioner.flatMap(_ match {
      case p: SingleDimensionPartitioner => Some(p)
      case _                             => None
    })

  def blocksPerPartition: Option[Int] =
    partitioner.map(_.dimensionSizePerPartition)
  def linesPerPartition: Option[Int] = blocksPerPartition.map(_ * blockSize)

  /**
    * Compute the matrix product A*B where A is this matrix, and B is a dense matrix
    * @param singleBlock: the B matrix by which we want to multiply
    * @return the multiplication result with SkinnyBlockMatrix format and same
    * partitioning as the initial object.
    */
  def singleBlockMultiply(singleBlock: BDM[Double],
                          persistResult: Boolean): SkinnyBlockMatrix = {
    require(
      numCols == singleBlock.rows,
      "The number of columns of A and the number of rows " +
        s"of B must be equal. A.numCols: $numCols, B.numRows: ${singleBlock.rows}. If you " +
        "think they should be equal, try setting the dimensions of A and B explicitly while " +
        "initializing them."
    )

    val broadcastSingleBlock = blocks.context.broadcast(singleBlock)
    val newBlocks = blocks.mapValues { block =>
      block * broadcastSingleBlock.value
    }

    val persistedBlocks =
      if (!persistResult) {
        newBlocks
      } else {
        val rdd = newBlocks.persist(StorageLevel.MEMORY_AND_DISK)
        rdd.count() //forcing compute to force persist
        rdd
      }

    SkinnyBlockMatrix(blocks = persistedBlocks,
                      blockSize = blockSize,
                      numRows = numRows,
                      numCols = singleBlock.cols,
                      partitionHeightInBlocks = partitionHeightInBlocks)
  }

  /**
    * Computes the dot product of the skinny block matrix u with another SkinnyBlockMatrix v
    * i.e. u`^`T v
    * @return Breeze Dense Matrix of size numCols x numCols
    */
  def dot(v: SkinnyBlockMatrix): BDM[Double] = {
    blocks
      .zipPartitions(v.blocks) {
        case (itU, itV) =>
          val result = BDM.zeros[Double](numCols, v.numCols)
          itU
            .zip(itV)
            .foreach({
              case ((indexU, uBlock), (indexV, vBlock)) =>
                assert(indexU == indexV)
                cforRange(0 until numCols) { i =>
                  cforRange(0 until v.numCols) { j =>
                    cforRange(0 until uBlock.rows) { k =>
                      result(i, j) += uBlock(k, i) * vBlock(k, j)
                    }
                  }
                }
            })
          Iterator(result)

      }
      .reduce(_ + _)
  }

  /**
    * Returns a new SkinnyBlockMatrix that is subtracting the argument from this object
    * @param x the matrix to substract
    * @return this - x
    */
  def minus(x: SkinnyBlockMatrix): SkinnyBlockMatrix = {
    assert(numRows == x.numRows)
    assert(numCols == x.numCols)
    assert(partitionHeightInBlocks == x.partitionHeightInBlocks)
    val data = blocks.zipPartitions(x.blocks) {
      case (it, itX) =>
        val mappedBlocks = itX.toMap
        it.map({
          case (i, block) =>
            assert(mappedBlocks.keySet.contains(i))
            (i, block - mappedBlocks(i))
        })
    }
    SkinnyBlockMatrix(data,
                      blockSize,
                      numRows,
                      numCols,
                      partitionHeightInBlocks)
  }

  /**
    * Do the QR decomposition of this matrix
    * @return a pair containing the orthogonal Q matrix and R which is a upper-triangular local matrix
    */
  def qr: (SkinnyBlockMatrix, BDM[Double]) = {
    require(
      numCols <= blockSize,
      s"QR-decomposition requires the number of columns to be at least " +
        s"as large as number of rows in a block. The number of columns is $numCols and the number of rows in" +
        s"a block is $blockSize"
    )

    val qrs = blocks.map {
      case (_, block) =>
        breeze.linalg.qr.reduced(block).r
    }

    // combine the R part from previous results vertically into a tall matrix
    // NB the order of the reduction is not guaranteed but it still gives a valid R matrix
    val reduceRsFun: (BDM[Double], BDM[Double]) => BDM[Double] = {
      case (r1, r2) =>
        val stackedR = BDM.vertcat(r1, r2)
        breeze.linalg.qr.reduced(stackedR).r
    }

    val combinedR = qrs.treeReduce(reduceRsFun, depth = 2)

    val invR = inv(combinedR)
    val finalQ = singleBlockMultiply(invR, persistResult = true)
    (finalQ, combinedR)
  }

  /**
    * Transform this matrix into an RDD of lines
    * @return a RDD of lines of the matrix
    */
  def toIndexedEmbeddings: RDD[(Long, BDV[Double])] = {
    blocks.flatMap {
      case (blockRowIdx, mat) =>
        Utils.rowIter(mat).zipWithIndex.map {
          case (vector, rowIdx) =>
            (1L * blockRowIdx * blockSize + rowIdx) -> vector
        }
    }
  }

  /**
    * Transfer this BlockMatrix into a dense breeze matrix (on the driver)
    * @return the same matrix but available on the driver
    */
  def toLocalMatrix: BDM[Double] = {
    require(numRows < Int.MaxValue,
            "The number of rows of this matrix should be less than " +
              s"Int.MaxValue. Currently numRows: $numRows")

    require(
      numRows * numCols < Int.MaxValue,
      "The length of the values array must be " +
        s"less than Int.MaxValue. Currently numRows * numCols: ${numRows * numCols}")

    val sortedBlocks = blocks
      .collect()
      .sortBy(_._1)
      .map(_._2)

    BDM.vertcat(sortedBlocks: _*)
  }

  /**
    * Repartition the blocks of this matrix by a new partitioner.
    * @param partitioner an instance of SingleDimensionPartitioner that will be used to repartition this matrix
    * @return a new instance of SkinnyBlockMatrix that is partitioned by [[partitioner]]
    */
  def repartitionBy(
      partitioner: SingleDimensionPartitioner): SkinnyBlockMatrix = {
    val newBlocks = blocks.partitionBy(partitioner)
    SkinnyBlockMatrix(newBlocks,
                      blockSize,
                      numRows,
                      numCols,
                      partitioner.dimensionSizePerPartition)
  }

}
