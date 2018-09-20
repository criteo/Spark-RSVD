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

import breeze.linalg.{diag, max, svd, DenseVector => BDV}
import com.criteo.rsvd.ReadingWritingData.RandomizedSVDKryoRegistrator
import com.google.common.math.LongMath
import com.google.common.primitives.Ints
import org.apache.spark.mllib.distributed.BlockMatrixTest
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.scalatest.FunSuite

object ReconstructionErrorTest {
  val expectedMachinePrecisionWithDouble: Double = 1e-12
}

class ReconstructionErrorTest extends FunSuite with PerTestSparkSession {
  import ReconstructionErrorTest._

  override def sparkConf
    : Map[String, Any] = // add override with PerTestSparkSession
    Map(
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryo.registrator" ->
        (RandomizedSVDKryoRegistrator.getClass.getName)
    )

  test("computeReconstructionError should give zero error on a rebuilt matrix") {
    val matSize = 205
    val blockSize = 20
    val embeddingDim = 10

    val partitionHeightInBlocks = 2
    val partitionWidthInBlocks = 3

    val u = SkinnyBlockMatrix.randomMatrix(matSize,
                                           embeddingDim,
                                           blockSize,
                                           partitionWidthInBlocks,
                                           sc,
                                           1234)
    val v = SkinnyBlockMatrix.randomMatrix(matSize,
                                           embeddingDim,
                                           blockSize,
                                           partitionWidthInBlocks,
                                           sc,
                                           12345)
    val sigma = BDV.ones[Double](embeddingDim)

    val localMat = u.toLocalMatrix * diag(sigma) * v.toLocalMatrix.t

    val entries =
      sc.parallelize(
        localMat.iterator
          .map({ case ((i, j), v) => MatrixEntry(i, j, v) })
          .toList)

    val error =
      ReconstructionError.computeReconstructionErrorFromEntries(u,
                                                                sigma,
                                                                v,
                                                                entries,
                                                                2.0)

    assert(error === 0)
  }

  test(
    "generateEntries with no negative sampling should give back the original matrix") {
    val matSize = 205
    val blockSize = 20
    val embeddingDim = 10

    val partitionHeightInBlocks = 2
    val partitionWidthInBlocks = 3

    val mat = BlockMatrixTest.createRandomMatrix(matSize,
                                                 matSize,
                                                 blockSize,
                                                 partitionHeightInBlocks,
                                                 partitionWidthInBlocks,
                                                 sc)

    val blockedEntries =
      ReconstructionError.generateBlockedEntries(mat, 0, 1234, true)
    val localBlocked = blockedEntries.collect()

    val entries = ReconstructionError.generateEntries(mat, 0, 1234, true)

    val mat2 = BlockMatrix.fromMatrixEntries(entries,
                                             matSize,
                                             matSize,
                                             blockSize,
                                             partitionHeightInBlocks,
                                             partitionWidthInBlocks)

    val entriesLoc = entries.collect()

    val localMat = mat.toLocalMatrix
    val localMat2 = mat2.toLocalMatrix

    assert(localMat === localMat2)
  }

  test("timesSVD should work for both direct and transpose operation") {
    val matHeight = 20
    val matWidth = 35
    val blockSize = 15
    val partitionHeightInBlocks = 1
    val partitionWidthInBlocks = 1

    val numDimBlocksHeight =
      Ints.checkedCast(
        LongMath.divide(matHeight, blockSize, RoundingMode.CEILING))

    val numDimBlocksWidth =
      Ints.checkedCast(
        LongMath.divide(matWidth, blockSize, RoundingMode.CEILING))

    val partitionerHeight =
      SingleDimensionPartitioner(numDimBlocksHeight, partitionHeightInBlocks)
    val partitionerWidth =
      SingleDimensionPartitioner(numDimBlocksWidth, partitionWidthInBlocks)

    val uDim = 7

    val u =
      SkinnyBlockMatrix.randomMatrix(matHeight,
                                     uDim,
                                     blockSize,
                                     partitionHeightInBlocks,
                                     sc,
                                     123556)
    val uLocal = u.toLocalMatrix

    val v =
      SkinnyBlockMatrix.randomMatrix(matWidth,
                                     uDim,
                                     blockSize,
                                     partitionWidthInBlocks,
                                     sc,
                                     4566789)
    val vLocal = v.toLocalMatrix

    val sigma = BDV.tabulate[Double](uDim) { i =>
      i
    }

    val x =
      SkinnyBlockMatrix
        .randomMatrix(matWidth,
                      uDim,
                      blockSize,
                      partitionWidthInBlocks,
                      sc,
                      6789)
    val xLocal = x.toLocalMatrix

    val y = SkinnyBlockMatrix
      .randomMatrix(matHeight,
                    uDim,
                    blockSize,
                    partitionHeightInBlocks,
                    sc,
                    89012)
    val yLocal = y.toLocalMatrix

    assert(
      Utils.relativeNormError(
        ReconstructionError
          .timesSVD(u, sigma, v, x, false)
          .toLocalMatrix,
        uLocal * diag(sigma) * vLocal.t * xLocal) < expectedMachinePrecisionWithDouble)

    assert(
      Utils.relativeNormError(
        ReconstructionError
          .timesSVD(u, sigma, v, y, true)
          .toLocalMatrix,
        vLocal * diag(sigma) * uLocal.t * yLocal) < expectedMachinePrecisionWithDouble)

  }

  test("findGoodBasisRSVDError gives a correct basis given enough iterations") {
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

    val uDim = 7

    val u =
      SkinnyBlockMatrix.randomMatrix(matSize,
                                     uDim,
                                     blockSize,
                                     partitionHeightInBlocks,
                                     sc,
                                     123556)
    val uLocal = u.toLocalMatrix

    val v =
      SkinnyBlockMatrix.randomMatrix(matSize,
                                     uDim,
                                     blockSize,
                                     partitionWidthInBlocks,
                                     sc,
                                     4566789)
    val vLocal = v.toLocalMatrix

    val sigma = BDV.ones[Double](uDim)

    val modifiedMat = mat.toLocalMatrix - uLocal * diag(sigma) * vLocal.t
    val svd.SVD(_, singularValModifiedMat, _) = svd(modifiedMat)

    val basisSize = 5
    val oversample = 10
    val numPowerIter = 15

    val randomBasis = SkinnyBlockMatrix.randomMatrix(matSize,
                                                     basisSize + oversample,
                                                     blockSize,
                                                     partitionWidthInBlocks,
                                                     sc,
                                                     348908190)

    val basis =
      ReconstructionError.findGoodBasisRSVDError(mat,
                                                 mat.transpose,
                                                 u,
                                                 sigma,
                                                 v,
                                                 randomBasis,
                                                 numPowerIter)

    val localBasis = basis.toLocalMatrix(::, 0 until basisSize)

    val residualMatrix = modifiedMat - localBasis * localBasis.t * modifiedMat

    val svd.SVD(_, singularValResidualMatrix, _) = svd(residualMatrix)

    assert(
      math.abs(max(singularValResidualMatrix) - singularValModifiedMat(
        basisSize)) < 2e-4) // converges very slowly !

  }
}
