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

import breeze.linalg.{diag, max, sum, svd, DenseMatrix => BDM}
import com.criteo.rsvd.ReadingWritingData.RandomizedSVDKryoRegistrator
import com.google.common.math.IntMath
import org.apache.spark.mllib.distributed.BlockMatrixTest
import org.scalatest.FunSuite

object RSVDTest {
  val expectedMachinePrecisionWithDouble: Double = 1e-12
  val matSize = 94
  val blockSize = 10
  val partitionHeightInBlocks = 2
  val partitionWidthInBlocks = 2
}

class RSVDTest extends FunSuite with PerTestSparkSession {
  import RSVDTest._

  override def sparkConf
    : Map[String, Any] = // add override with PerTestSparkSession
    Map(
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryo.registrator" -> RandomizedSVDKryoRegistrator.getClass.getName
    )

  test("findGoodBasis is within theoretical (tight) approximation bounds") {
    val mat = BlockMatrixTest.createRandomMatrix(matHeight = matSize,
                                                 matWidth = matSize,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc)

    val dim = 5
    val oversample = 1
    val seed = 0
    val randomMat = SkinnyBlockMatrix.randomMatrix(matSize,
                                                   dim + oversample,
                                                   blockSize,
                                                   partitionWidthInBlocks,
                                                   sc,
                                                   seed)

    val numPowerIterations = 10
    val basis = RSVD
      .findGoodBasis(mat, mat.transpose, randomMat, numPowerIterations)
      .toLocalMatrix

    val svd.SVD(_, s, _) = svd(mat.toLocalMatrix)

    /*
    Estimating upper error bound as in Corollary 1.5 in:
    Randomized Algorithms for Low-Rank Matrix Factorizations:
    Sharp Performance Bounds
    https://arxiv.org/pdf/1308.5697.pdf
     */
    var upperBoundMultiplier = Math.sqrt(matSize - dim) + Math.sqrt(dim)
    upperBoundMultiplier = upperBoundMultiplier / (Math.sqrt(dim + oversample) - Math
      .sqrt(dim))
    upperBoundMultiplier =
      Math.pow(upperBoundMultiplier, 1.0 / (1 + 2 * numPowerIterations))

    val sTargetRankPlusOne = s(dim)
    val upperBound = sTargetRankPlusOne * upperBoundMultiplier

    // Computing the residual norm
    val eye = BDM.eye[Double](matSize.toInt)
    val residualMatrix = (eye - basis * basis.t) * mat.toLocalMatrix
    val svd.SVD(_, residualSingularValues, _) = svd(residualMatrix)
    val residualSpectralNorm = max(residualSingularValues)
    assert(residualSpectralNorm <= upperBound)

    // With enough power iterations and oversampling we should approximate better than k top singular values
    assert(sum(residualSingularValues) <= sum(s(dim until matSize)))
  }

  test(
    "computeSingularVectors recover original singular vectors given perfect basis") {
    val numDimBlocks = IntMath.divide(matSize, blockSize, RoundingMode.CEILING)
    val mat = BlockMatrixTest.createRandomMatrix(matHeight = matSize,
                                                 matWidth = matSize,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc)

    val svd.SVD(u, s, vt) = svd.reduced(mat.toLocalMatrix)
    val v = vt.t
    Utils.deterministicSignsInplace(u, v)
    val targetRank = 5
    val perfectBasis = u(::, 0 until targetRank)
    val blocks = sc
      .parallelize((0 until numDimBlocks).map { idx =>
        idx -> perfectBasis(
          idx * blockSize until Math.min((idx + 1) * blockSize, matSize),
          ::).copy
      })

    val perfectSkinnyBasis =
      SkinnyBlockMatrix(blocks,
                        blockSize,
                        matSize,
                        targetRank,
                        partitionHeightInBlocks)

    val RsvdResults(leftSingularVectors, singularValues, _) =
      RSVD.computeSingularVectors(mat.transpose,
                                  perfectSkinnyBasis,
                                  targetRank,
                                  computeLeftSingularVectors = true,
                                  computeRightSingularVectors = false)

    val localMat = mat.toLocalMatrix
    val reconstructedBasis = leftSingularVectors.get.toLocalMatrix
    val projectionToTargetSingularVectors = perfectBasis * perfectBasis.t * localMat
    val projectionToApproxSingularVectors = reconstructedBasis * reconstructedBasis.t * localMat
    val errorVecs =
      Utils.relativeNormError(projectionToTargetSingularVectors,
                              projectionToApproxSingularVectors)
    assert(errorVecs <= expectedMachinePrecisionWithDouble)

    val errorValues =
      Utils.relativeNormError(s(0 until targetRank).copy, singularValues)
    assert(errorValues <= expectedMachinePrecisionWithDouble)
  }

  test(
    "Given perfect basis, computeSingularVectors recovers the same approximation as a perfect SVD") {
    val numDimBlocks = IntMath.divide(matSize, blockSize, RoundingMode.CEILING)
    val mat = BlockMatrixTest.createRandomMatrix(matHeight = matSize,
                                                 matWidth = matSize,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc)

    val svd.SVD(u, s, vt) = svd.reduced(mat.toLocalMatrix)
    val v = vt.t
    Utils.deterministicSignsInplace(u, v)
    val targetRank = 5
    val perfectBasis = u(::, 0 until targetRank)
    val blocks = sc
      .parallelize((0 until numDimBlocks).map { idx =>
        idx -> perfectBasis(
          idx * blockSize until Math.min((idx + 1) * blockSize, matSize),
          ::).copy
      })

    val perfectSkinnyBasis =
      SkinnyBlockMatrix(blocks,
                        blockSize,
                        matSize,
                        targetRank,
                        partitionWidthInBlocks)
    val RsvdResults(leftSingularVectors, singularValues, rightSingularVectors) =
      RSVD.computeSingularVectors(mat.transpose,
                                  perfectSkinnyBasis,
                                  targetRank,
                                  computeLeftSingularVectors = true,
                                  computeRightSingularVectors = true)

    val leftSingularVectorsLocal = leftSingularVectors.get.toLocalMatrix
    val rightSingularVectorsLocal = rightSingularVectors.get.toLocalMatrix

    Utils.deterministicSignsInplace(leftSingularVectorsLocal,
                                    rightSingularVectorsLocal)

    val truncatedU = u(::, 0 until targetRank)
    val truncatedV = v(::, 0 until targetRank)
    val truncatedSingVal = s(0 until targetRank)
    val SVDApproximation = truncatedU * diag(truncatedSingVal) * truncatedV.t

    val RSVDApproximation = leftSingularVectors.get.toLocalMatrix * diag(
      singularValues) * rightSingularVectors.get.toLocalMatrix.t

    val error = Utils.relativeNormError(SVDApproximation, RSVDApproximation)

    assert(error <= expectedMachinePrecisionWithDouble)

    val leftSingularVecError =
      Utils.relativeNormError(truncatedU, leftSingularVectorsLocal)
    val rightSingularVecError =
      Utils.relativeNormError(truncatedV, rightSingularVectorsLocal)

    assert(leftSingularVecError <= expectedMachinePrecisionWithDouble)
    assert(rightSingularVecError <= expectedMachinePrecisionWithDouble)

  }

  test(
    "findGoodBasis + computeLeftSingularVectors recover original left singular vectors") {
    val matSize = 30 // Smaller matrix to get close to perfect result
    val blockSize = 15
    val partitionHeightInBlocks = 1
    val partitionWidthInBlocks = 1

    val mat = BlockMatrixTest.createRandomMatrix(matHeight = matSize,
                                                 matWidth = matSize,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc)

    val svd.SVD(_, s, _) = svd(mat.toLocalMatrix)

    val dim = 5
    val oversample = 10 // Has to be large to get close to perfect result
    val seed = 0
    val randomMat = SkinnyBlockMatrix.randomMatrix(matSize,
                                                   dim + oversample,
                                                   blockSize,
                                                   partitionWidthInBlocks,
                                                   sc,
                                                   seed)

    val numPowerIterations = 10
    val basis =
      RSVD.findGoodBasis(mat, mat.transpose, randomMat, numPowerIterations)
    val RsvdResults(u2, _, _) =
      RSVD.computeSingularVectors(mat.transpose,
                                  basis,
                                  dim,
                                  computeLeftSingularVectors = true,
                                  computeRightSingularVectors = false)
    val u2Local = u2.get.toLocalMatrix

    // Computing the residual norm
    val eye = BDM.eye[Double](matSize)
    val residualMatrix = (eye - u2Local * u2Local.t) * mat.toLocalMatrix
    val svd.SVD(_, residualSingularValues, _) = svd(residualMatrix)
    val residualSpectralNorm = max(residualSingularValues)
    assert(
      Math.abs(residualSpectralNorm - s(dim)) <= 10 * expectedMachinePrecisionWithDouble)
  }

  test(
    "findGoodBasis + computeLeftSingularVectors recover original left singular vectors for rectangular case") {
    val matHeight = 30 // Smaller matrix to get close to perfect result
    val matWidth = 60 // Smaller matrix to get close to perfect result
    val blockSize = 15
    val partitionHeightInBlocks = 1
    val partitionWidthInBlocks = 1

    val mat = BlockMatrixTest.createRandomMatrix(matHeight = matHeight,
                                                 matWidth = matWidth,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc)

    val svd.SVD(_, s, _) = svd(mat.toLocalMatrix)

    val dim = 5
    val oversample = 10 // Has to be large to get close to perfect result
    val seed = 0
    val randomMat = SkinnyBlockMatrix.randomMatrix(matWidth,
                                                   dim + oversample,
                                                   blockSize,
                                                   partitionWidthInBlocks,
                                                   sc,
                                                   seed)

    val numPowerIterations = 10
    val basis =
      RSVD.findGoodBasis(mat, mat.transpose, randomMat, numPowerIterations)
    val RsvdResults(u2, _, _) =
      RSVD.computeSingularVectors(mat.transpose,
                                  basis,
                                  dim,
                                  computeLeftSingularVectors = true,
                                  computeRightSingularVectors = false)
    val u2Local = u2.get.toLocalMatrix

    // Computing the residual norm
    val eye = BDM.eye[Double](matHeight)
    val residualMatrix = (eye - u2Local * u2Local.t) * mat.toLocalMatrix
    val svd.SVD(_, residualSingularValues, _) = svd(residualMatrix)
    val residualSpectralNorm = max(residualSingularValues)
    assert(
      Math.abs(residualSpectralNorm - s(dim)) <= 10 * expectedMachinePrecisionWithDouble)
  }

}
