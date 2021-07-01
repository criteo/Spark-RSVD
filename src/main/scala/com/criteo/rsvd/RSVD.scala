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

import breeze.linalg.svd.SVD
import breeze.linalg.{svd, DenseVector => BDV}
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext

case class RSVDConfig(
    embeddingDim: Int = 100,
    oversample: Int = 30,
    powerIter: Int = 1,
    seed: Int = 0,
    blockSize: Int = 50000,
    partitionWidthInBlocks: Int = 35,
    partitionHeightInBlocks: Int = 10,
    computeLeftSingularVectors: Boolean = true,
    computeRightSingularVectors: Boolean = false
)

case class RsvdResults(leftSingularVectors: Option[SkinnyBlockMatrix],
                       singularValues: BDV[Double],
                       rightSingularVectors: Option[SkinnyBlockMatrix])

object RSVD extends StrictLogging {

  /**
    * Launch the R-SVD decomposition of a given matrix.
    * @param inputMatrix the matrix to run RSVD on
    * @param config RSVD configuration parameters
    * @param sc spark context
    * @return a object containing the RSVD decomposition
    */
  def run(inputMatrix: BlockMatrix,
          config: RSVDConfig,
          sc: SparkContext): RsvdResults = {

    if (inputMatrix.blockSize != config.blockSize) {
      throw new IllegalArgumentException(
        s"Expected a matrix blocksize of ${config.blockSize}, got ${inputMatrix.blockSize}")
    }

    if (inputMatrix.partitionWidthInBlocks != config.partitionWidthInBlocks) {
      throw new IllegalArgumentException(
        s"Expected a matrix partitionWidthInBlocks of ${config.partitionWidthInBlocks}, got ${inputMatrix.partitionWidthInBlocks}")
    }

    if (inputMatrix.partitionHeightInBlocks != config.partitionHeightInBlocks) {
      throw new IllegalArgumentException(
        s"Expected a matrix partitionHeightInBlocks of ${config.partitionHeightInBlocks}, got ${inputMatrix.partitionHeightInBlocks}")
    }

    // compute size in MB of the partitions of the Dense Matrix:
    val nEntriesInDensePartition = (config.embeddingDim + config.oversample) * config.blockSize * config.partitionHeightInBlocks
    val partitionSizeMB =
      (nEntriesInDensePartition * 8 / (1024.0 * 1024.0)).toInt

    if (partitionSizeMB >= 2000) {
      logger.error(
        s"Dense matrix partition size ($partitionSizeMB MB) is above 2GB, which will probably make the job fail if they end up persisted on disk (cf SPARK-3151)")
    } else if (partitionSizeMB >= 1000) {
      logger.warn(
        s"Dense matrix partition size is $partitionSizeMB MB, which may be too high")
    }

    val matrix = inputMatrix.persist(StorageLevel.DISK_ONLY)

    val matTranspose = matrix.transpose
      .persist(StorageLevel.DISK_ONLY)

    sc.setJobDescription("Producing random projection basis")
    val projectionDimension = config.embeddingDim + config.oversample
    val randomMat = SkinnyBlockMatrix.randomMatrix(
      inputMatrix.matWidth,
      projectionDimension,
      config.blockSize,
      config.partitionWidthInBlocks,
      sc,
      config.seed)
    val basis = findGoodBasis(matrix, matTranspose, randomMat, config.powerIter)

    sc.setJobDescription("Computing left singular vectors")

    computeSingularVectors(
      matTranspose,
      basis,
      config.embeddingDim,
      computeLeftSingularVectors = config.computeLeftSingularVectors,
      computeRightSingularVectors = config.computeRightSingularVectors
    )
  }

  /**
    * findGoodBasis implements power iteration with orthogonalization in between
    * multiplications for better numerical stability. The final orthogonalization
    * is to achieve as good orthogonalization as possible because it will directly
    * impact the quality of our basis.
    * The goal of this method is to find a basis that would capture most of the
    * norm in the SquareMatrix and would be orthogonal to the lower part of the
    * spectrum.
    */
  private[rsvd] def findGoodBasis(mat: BlockMatrix,
                                  matTranspose: BlockMatrix,
                                  randomMat: SkinnyBlockMatrix,
                                  powerIter: Int): SkinnyBlockMatrix = {

    var basis = mat
      .skinnyMultiply(randomMat, repartitionAndPersistSkinnyMatrix = false)
      .qr
      ._1

    (0 until powerIter).foreach { iterIdx =>
      randomMat.blocks.context
        .setJobDescription(s"${iterIdx + 1} power iteration")
      basis = matTranspose
        .skinnyMultiply(basis, repartitionAndPersistSkinnyMatrix = true)
        .qr
        ._1
      basis = mat
        .skinnyMultiply(basis, repartitionAndPersistSkinnyMatrix = true)
        .qr
        ._1
    }

    basis.qr._1
  }

  /**
    * Given a good basis recovers almost perfectly the original left singular vectors from A.
    * The error can be reduced by using larger oversampling in the basis.
    */
  private[rsvd] def computeSingularVectors(
      matTranspose: BlockMatrix,
      basis: SkinnyBlockMatrix,
      targetRank: Int,
      computeLeftSingularVectors: Boolean,
      computeRightSingularVectors: Boolean): RsvdResults = {
    val resultOfProjection =
      matTranspose.skinnyMultiply(basis,
                                  repartitionAndPersistSkinnyMatrix = true)
    val (q, r) = resultOfProjection.qr
    val SVD(u, s, vt) = svd.reduced(r)
    val v = vt.t
    Utils.deterministicSignsInplace(u, v)

    val singularValues = s(0 until targetRank)

    //check the decomposition is indeed by decreasing singular values
    if (singularValues.length >= 2) {
      (0 until singularValues.length - 1).foreach { i =>
        require(singularValues.data(i) >= singularValues.data(i + 1))
      }
    }

    val leftSingularVectors = computeLeftSingularVectors match {
      case true =>
        Some(
          basis.singleBlockMultiply(v(::, 0 until targetRank),
                                    persistResult = true))
      case false => None
    }

    val rightSingularVectors = computeRightSingularVectors match {
      case true =>
        Some(
          q.singleBlockMultiply(u(::, 0 until targetRank),
                                persistResult = true))
      case false => None
    }

    RsvdResults(leftSingularVectors, singularValues, rightSingularVectors)
  }

}
