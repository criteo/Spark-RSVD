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

import breeze.linalg.svd.SVD
import breeze.linalg.{diag, max, svd, DenseMatrix => BDM, DenseVector => BDV}
import com.google.common.math.LongMath
import com.google.common.primitives.Ints
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.rdd.RDD
import spire.syntax.cfor.cforRange

import scala.util.Random

object ReconstructionError extends StrictLogging {

  private[rsvd] def timesSVD(u: SkinnyBlockMatrix,
                             sigma: BDV[Double],
                             v: SkinnyBlockMatrix,
                             x: SkinnyBlockMatrix,
                             isTransposed: Boolean): SkinnyBlockMatrix = {
    if (!isTransposed) {
      val sigmaVx = diag(sigma) * v.dot(x)
      u.singleBlockMultiply(sigmaVx, persistResult = false)
    } else {
      val sigmaUx = diag(sigma) * u.dot(x)
      v.singleBlockMultiply(sigmaUx, persistResult = false)
    }
  }

  /**
    * findGoodBasisRSVDError implements a power iteration just as findGoodBasis, adding a rank k update
    * to the matrix so as to compute the basis of A - U * Sigma * V`^`T, where U, Sigma, V is the partial SVD decomposition;
    * It enables the computation of the 2-norm of the matrix A - U * Sigma * V`^`T, so as to compute the reconstruction
    * error of the SVD decomposition
    * @param mat: BlockMatrix. Matrix A
    * @param matTranspose: BlockMatrix. Matrix A`^`T
    * @param u: SkinnyBlockMatrix. Matrix U
    * @param sigma: BreezeDenseVector[Double]. Singular values
    * @param v: SkinnyBlockMatrix. Matrix V
    * @param randomMat: SkinnyBlockMatrix. Random matrix to initiate the power iterations
    * @param powerIter: Int. Number of power iterations for the iterative algorithm
    * @return the basis needed for the SVD decomposition of A - U * Sigma * V`^`T
    */
  private[rsvd] def findGoodBasisRSVDError(
      mat: BlockMatrix,
      matTranspose: BlockMatrix,
      u: SkinnyBlockMatrix,
      sigma: BDV[Double],
      v: SkinnyBlockMatrix,
      randomMat: SkinnyBlockMatrix,
      powerIter: Int): SkinnyBlockMatrix = {

    assert(u.numRows == mat.matHeight)
    assert(v.numRows == mat.matHeight)
    assert(u.numCols == v.numCols)
    assert(u.numCols == sigma.length)

    val partitionerMatrixHeight =
      SingleDimensionPartitioner(mat.numDimBlocksHeight,
                                 mat.partitionWidthInBlocks)
    val partitionerMatrixWidth =
      SingleDimensionPartitioner(mat.numDimBlocksWidth,
                                 mat.partitionWidthInBlocks)

    val repartitionedU = u.repartitionBy(partitionerMatrixHeight)
    val repartitionedV = v.repartitionBy(partitionerMatrixWidth) //useless if we use the new partitioner checking function !

    val repartitionedUt = u.repartitionBy(partitionerMatrixWidth)
    val repartitionedVt = v.repartitionBy(partitionerMatrixHeight)

    var basis = mat
      .skinnyMultiplyMinusSVD(randomMat,
                              repartitionedU,
                              sigma,
                              repartitionedV,
                              repartitionAndPersistSkinnyMatrix = true,
                              false)
      .qr
      ._1

    (0 until powerIter).foreach({ iterIndex =>
      basis = matTranspose
        .skinnyMultiplyMinusSVD(basis,
                                repartitionedUt,
                                sigma,
                                repartitionedVt,
                                repartitionAndPersistSkinnyMatrix = true,
                                true)
        .qr
        ._1

      basis = mat
        .skinnyMultiplyMinusSVD(basis,
                                repartitionedU,
                                sigma,
                                repartitionedV,
                                repartitionAndPersistSkinnyMatrix = true,
                                false)
        .qr
        ._1
    })

    basis.qr._1
  }

  private[rsvd] def computeSingularValueRSVDResidual(
      matTranspose: BlockMatrix,
      basis: SkinnyBlockMatrix,
      u: SkinnyBlockMatrix,
      sigma: BDV[Double],
      v: SkinnyBlockMatrix,
      targetRank: Int): BDV[Double] = {
    val partitionerMatrixHeight = SingleDimensionPartitioner(
      matTranspose.numDimBlocksWidth,
      matTranspose.partitionWidthInBlocks)

    val repartitionedU = u.repartitionBy(partitionerMatrixHeight)
    val repartitionedV = v.repartitionBy(partitionerMatrixHeight)

    val correction =
      timesSVD(repartitionedU, sigma, repartitionedV, basis, false)
    val resultOfProjection =
      matTranspose
        .skinnyMultiply(basis, repartitionAndPersistSkinnyMatrix = true)
        .repartitionBy(partitionerMatrixHeight)
    val correctedBasis = resultOfProjection.minus(correction)
    val (_, r) = correctedBasis.qr
    val SVD(_, s, _) = svd.reduced(r)
    s(0 until targetRank)
  }

  def computeReconstructionErrorByRSVD(mat: BlockMatrix,
                                       matTranspose: BlockMatrix,
                                       u: SkinnyBlockMatrix,
                                       sigma: BDV[Double],
                                       v: SkinnyBlockMatrix,
                                       config: RSVDConfig): Double = {

    val randomBasis = SkinnyBlockMatrix.randomMatrix(
      mat.matHeight,
      config.embeddingDim + config.oversample,
      config.blockSize,
      config.partitionWidthInBlocks,
      u.blocks.context,
      config.seed + 1234
    )

    val basis =
      findGoodBasisRSVDError(mat,
                             matTranspose,
                             u,
                             sigma,
                             v,
                             randomBasis,
                             config.powerIter)

    val singularValues =
      computeSingularValueRSVDResidual(matTranspose,
                                       basis,
                                       u,
                                       sigma,
                                       v,
                                       config.embeddingDim)
    max(singularValues)
  }

  /**
    * Generate entries to be used by computeNormPartition.
    * All the entries of the given matrix are listed in the entries,
    * and also one random entry for each "true" entry of the matrix.
    * To simplify the code, the random entries are generated in each partition
    * of the local matrix. Thus, the distribution of the random entries is not
    * uniform, but follows the distribution of the matrix entries.
    *
    * @param mat: SquareBlockMatrix from which some entries will be generated
    * @param negativeEntriesPerPositiveEntries: Int. How many randomly sampled "negative" entries are sampled per existing entry in the matrix
    * @param seed: Long. Seed to generate the random entries
    * @param preciseVerification: Boolean. If true, full check if the random entry is a false negative, i.e. already exists
    *                           in the matrix, else puts 0 if it exists or not.
    * @return RDD of matrix entries, already collected into blocks (see also SquareBlockMatrix.fromMatrixEntries)
    */
  private[rsvd] def generateBlockedEntries(
      mat: BlockMatrix,
      negativeEntriesPerPositiveEntries: Int,
      seed: Long,
      preciseVerification: Boolean)
    : RDD[((Int, Int), Iterator[MatrixEntry])] = {

    val iteratorsTrueEntries = mat.blocksRDDs.map(rdd =>
      rdd.map {
        case ((i, j), csr) =>
          ((i, j), csr.activeIterator.map({
            case ((i, j), v) => MatrixEntry(i, j, v)
          }))
    })

    val numDimBlocksWidth =
      Ints.checkedCast(
        LongMath.divide(mat.matWidth, mat.blockSize, RoundingMode.CEILING))

    val iteratorsRandomEntries = mat.blocksRDDs.map(rdd =>
      rdd.map {
        case ((i, j), csr) =>
          val random = new Random(seed + (i + j * numDimBlocksWidth).toLong)
          val res = new Array[MatrixEntry](
            csr.data.length * negativeEntriesPerPositiveEntries)
          cforRange(0 until res.length) { k =>
            val ii = random.nextInt(csr.rows)
            val jj = random.nextInt(csr.cols)
            val v = if (preciseVerification) csr(ii, jj) else 0.0 // get the value if it already exists
            res(k) = MatrixEntry(ii, jj, v)
          }
          ((i, j), res.iterator)
    })

    val sc = mat.blocksRDDs(0).sparkContext
    val trueEntries = sc.union(iteratorsTrueEntries)
    val randomEntries = sc.union(iteratorsRandomEntries)

    sc.union(Seq(trueEntries, randomEntries))
  }

  /**
    * generateRandomEntries take a BlockMatrix mat, and outputs and RDD of MatrixEntry, which contains all entries
    * of the matrix mat, and some randomly sampled entries in the matrix which are supposed to be negative (i.e. zero).
    * This last approximation entails some error, but for very sparse matrices, the impact on the evaluated reconstruction
    * error should be insignificant.
    * @param mat : BlockMatrix. Matrix from which the positive entries are sampled.
    * @param negativeEntriesPerPositiveEntries : Int. Number of negative entries randomly generated per positive entry of the matrix
    * @param seed: Long. Seed to initialise the random generation of coordinates.
    * @return
    */
  private[rsvd] def generateRandomEntries(
      mat: BlockMatrix,
      negativeEntriesPerPositiveEntries: Int,
      seed: Long): RDD[MatrixEntry] = {

    val entriesArray = mat.blocksRDDs.map(rdd =>
      rdd.flatMap({
        case ((iBlock, jBlock), csr) =>
          csr.activeIterator.map({
            case ((i, j), v) =>
              MatrixEntry(iBlock * mat.blockSize + i,
                          jBlock * mat.blockSize + j,
                          v)
          })
      }))

    val blockSize = mat.blockSize
    val numDimBlocksHeight = mat.numDimBlocksHeight
    val numDimBlocksWidth = mat.numDimBlocksWidth // for serialization issue

    val widthLastBlock = mat.matWidth % mat.blockSize
    val heightLastBlock = mat.matHeight % mat.blockSize

    val randomEntriesArray = mat.blocksRDDs.map(rdd =>
      rdd.flatMap({
        case ((l, m), csr) =>
          val random = new Random(seed + (l + m * numDimBlocksWidth).toLong)
          Iterator[Int](negativeEntriesPerPositiveEntries * csr.data.length)
            .map({
              _ =>
                var succeeded = false
                var entry = MatrixEntry(0, 0, 0.0) // very messy
                while (!succeeded) {
                  val iBlock = random.nextInt(numDimBlocksWidth)
                  val jBlock = random.nextInt(numDimBlocksHeight)
                  val i = random.nextInt(blockSize)
                  val j = random.nextInt(blockSize)
                  succeeded =
                    !(((iBlock == numDimBlocksHeight - 1) && (i >= heightLastBlock)) || ((jBlock ==
                      numDimBlocksWidth - 1) && (j >= widthLastBlock)))
                  entry = MatrixEntry(iBlock * blockSize + i,
                                      jBlock * blockSize + j,
                                      0.0)
                }
                entry
            })
      }))

    val sc = mat.blocksRDDs(0).sparkContext

    sc.union((entriesArray.toIterator ++ randomEntriesArray.toIterator).toSeq)

  }

  private[rsvd] def generateEntries(
      mat: BlockMatrix,
      negativeEntriesPerPositiveEntries: Int,
      seed: Long,
      preciseVerification: Boolean): RDD[MatrixEntry] = {
    val blockedEntries =
      generateBlockedEntries(mat,
                             negativeEntriesPerPositiveEntries,
                             seed,
                             preciseVerification)

    val blockSize = mat.blockSize

    blockedEntries.flatMap({
      case ((i, j), it) =>
        it.map(
            entry =>
              MatrixEntry(entry.i + i * blockSize,
                          entry.j + j * blockSize,
                          entry.value))
          .toList
    })

  }

  private[rsvd] def computeNormPartition(
      blockedEntriesIt: Iterator[((Int, Int), Iterable[MatrixEntry])],
      blocksIterator: Iterator[((Int, BDM[Double]), (Int, BDM[Double]))],
      singularValues: BDV[Double],
      norm: (Double, Double) => Double
  ): Iterator[Double] = {
    val blockedEntriesMap = blockedEntriesIt.toMap
    var result: Double = 0.0
    blocksIterator.foreach {
      case ((rowIndex, u_i), (colIndex, v_j)) =>
        blockedEntriesMap.get((rowIndex, colIndex)) match {
          case Some(entriesIt) => {
            entriesIt.foreach({ entry =>
              var rebuiltValue = 0.0
              cforRange(0 until singularValues.length) { k =>
                rebuiltValue += u_i(entry.i.toInt, k) * singularValues(k) * v_j(
                  entry.j.toInt,
                  k)
              }
              result += norm(rebuiltValue, entry.value)
            })
          }
          case None => Unit
        }
    }
    Iterator(result)
  }

  def computeReconstructionErrorFromEntries(
      leftSingularVectors: SkinnyBlockMatrix,
      singularValues: BDV[Double],
      rightSingularVectors: SkinnyBlockMatrix,
      entriesRDD: RDD[MatrixEntry],
      norm: Double = 2.0): Double = {
    require(norm > 0.0)

    val normsRDD = computeReconstructionErrorRDD(leftSingularVectors,
                                                 singularValues,
                                                 rightSingularVectors,
                                                 entriesRDD,
                                                 norm)

    val (totalSum, totalCount) = normsRDD.map(x => (x, 1L)).reduce {
      case ((sum1, count1), (sum2, count2)) => (sum1 + sum2, count1 + count2)
    }

    math.pow(totalSum / totalCount, 1.0 / norm)
  }

  def computeReconstructionErrorRDD(leftSingularVectors: SkinnyBlockMatrix,
                                    singularValues: BDV[Double],
                                    rightSingularVectors: SkinnyBlockMatrix,
                                    entriesRDD: RDD[MatrixEntry],
                                    norm: Double = 2.0): RDD[Double] = {

    val nbEntries = entriesRDD.count()
    val nbEntriesPerPart = 2000000L //2M

    val nbPart = nbEntries / nbEntriesPerPart + 1

    logger.info(s"computeReconstructionError: using $nbPart partitions")
    logger.info(
      s"computeReconstructionError: Partition size: ${2 * nbEntriesPerPart * leftSingularVectors.numCols * 8 / (1024.0 * 1024.0)} MB")

    val blockSize = leftSingularVectors.blockSize

    val partitioner1 = new HashPartitioner(nbPart.toInt)
    val partitioner2 = new HashPartitioner(nbPart.toInt)

    val entriesForLeftSingular = entriesRDD
      .map { e =>
        (e.i, e)
      }
      .partitionBy(partitioner1)

    val leftSingularVectorsSplit = leftSingularVectors.blocks
      .flatMap {
        case (blockId, matrix) =>
          Utils.rowIter(matrix).zipWithIndex.map {
            case (row, idInBlock) =>
              ((idInBlock + blockId * blockSize).toLong, row)
          }
      }
      .partitionBy(partitioner1)

    //Only keep the left singular vectors that are actually required, but we also have
    // to keep the list of all coordinates and values requested for a given "i" index.
    val dispatchedRowVectors = leftSingularVectorsSplit
      .zipPartitions(entriesForLeftSingular) {
        case (singularVectors, entries) =>
          val iToEntries = entries.toList.groupBy(_._1)

          (for {
            (i, singularVector) <- singularVectors
          } yield {
            iToEntries.get(i) match {
              case None       => Iterator()
              case Some(list) => List((i, (singularVector, list.map(_._2))))
            }
          }).flatten

      }
      .partitionBy(partitioner2)

    val entriesForRightSingular = entriesRDD
      .map { e =>
        (e.j, e)
      }
      .partitionBy(partitioner1)

    val rightSingularVectorsSplit = rightSingularVectors.blocks
      .flatMap {
        case (blockId, matrix) =>
          Utils.rowIter(matrix).zipWithIndex.map {
            case (row, idInBlock) =>
              ((idInBlock + blockId * blockSize).toLong, row)
          }
      }
      .partitionBy(partitioner1)

    //Only keep the right singular vectors that are actually required for each i,
    // but we also don't send the same j vectors if they would end up in the same partition.
    val dispatchedColVectors = rightSingularVectorsSplit
      .zipPartitions(entriesForRightSingular) {
        case (singularVectors, coords) =>
          val jToEntries = coords.map(_._2).toList.groupBy(_.j)

          (for {
            (j, singularVector) <- singularVectors
          } yield {
            jToEntries.get(j) match {
              case None => Iterator()
              case Some(entries) =>
                val alreadySentSingularVectors =
                  scala.collection.mutable.SortedSet[Int]()

                entries.flatMap { entry =>
                  val partitionToDispatchVector =
                    partitioner2.getPartition(entry.i)

                  //if vector J has to be sent to several Is that are on the same partition, send it only once
                  if (alreadySentSingularVectors.contains(
                        partitionToDispatchVector)) {
                    None
                  } else {
                    alreadySentSingularVectors.add(partitionToDispatchVector)
                    Some((entry.i, (j, singularVector)))
                  }

                }
            }
          }).flatten

      }
      .partitionBy(partitioner2)

    dispatchedRowVectors
      .zipPartitions(dispatchedColVectors) {
        case (iVectors, jVectors) =>
          val jVectorsByJs = jVectors.map(_._2).toMap

          iVectors.flatMap {
            case (_, (embeddingI, entries)) =>
              entries.map { entry =>
                val embeddingJ = jVectorsByJs(entry.j)

                var res = 0.0
                (0 until singularValues.length).foreach { k =>
                  res += singularValues.data(k) * embeddingI.data(k) * embeddingJ
                    .data(k)
                }
                math.pow(math.abs(res - entry.value), norm)
              }

          }
      }
  }

  case class ReconstructionErrorDistribution(sampleSize: Int,
                                             average: Double,
                                             variance: Double,
                                             quantilesArray: Array[Double])

  def computeStatsForReconstructionError(
      leftSingularVectors: SkinnyBlockMatrix,
      singularValues: BDV[Double],
      rightSingularVectors: SkinnyBlockMatrix,
      entriesRDD0: RDD[MatrixEntry],
      norm: Double = 2.0,
      nbEntriesOnDriver: Long = 50000000L, //50M
      nQuantiles: Int = 100): ReconstructionErrorDistribution = {

    require(norm > 0.0)

    val nbEntries = entriesRDD0.count()

    val samplingRate =
      if (nbEntries < nbEntriesOnDriver) 1.0
      else
        nbEntriesOnDriver.toDouble / nbEntries

    logger.info(s"Sampling rate: ${samplingRate * 100.0}%")

    val sampledRDD = entriesRDD0.sample(false, samplingRate)

    val norms = computeReconstructionErrorRDD(leftSingularVectors,
                                              singularValues,
                                              rightSingularVectors,
                                              sampledRDD,
                                              norm)

    val sortedNorms = norms.collect().map(x => math.pow(x, 1 / norm)).sorted
    val size = sortedNorms.length

    val quantilesArray = new Array[Double](nQuantiles + 1)
    (0 until nQuantiles).foreach { i =>
      quantilesArray(i) = sortedNorms(((i * size.toLong) / nQuantiles).toInt)
    }
    quantilesArray(nQuantiles) = sortedNorms(size - 1)

    val average = sortedNorms.sum / sortedNorms.length

    var variance = 0.0
    sortedNorms.foreach { x =>
      variance += (x - average) * (x - average)
    }
    variance = variance / (size - 1)

    ReconstructionErrorDistribution(sampleSize = size,
                                    average = average,
                                    variance = variance,
                                    quantilesArray = quantilesArray)
  }

  def computeReconstructionError(leftSingularVector: SkinnyBlockMatrix,
                                 singularValues: BDV[Double],
                                 rightSingularVectors: SkinnyBlockMatrix,
                                 norm: Double,
                                 mat: BlockMatrix,
                                 seed: Long = 12345): Double = {
    val entries = generateEntries(mat, 1, seed, true)
    computeReconstructionErrorFromEntries(leftSingularVector,
                                          singularValues,
                                          rightSingularVectors,
                                          entries,
                                          norm)
  }
}
