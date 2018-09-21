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

import java.nio.file.{Files, Paths}

import org.apache.commons.io.FileUtils
import org.scalatest.FunSuite
import org.scalactic.Tolerance._
import com.google.common.io.Resources
import com.typesafe.scalalogging.slf4j.StrictLogging
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.storage.StorageLevel

class RandomizedSVDJobEndToEndTest
    extends FunSuite
    with StrictLogging
    with PerTestSparkSession {

  test("Test on real tiny dataset, with unconverged power iterations") {
    val tmpDir = Files.createTempDirectory("r-svd").toFile
    val inputPath = Paths
      .get(Resources.getResource("tiny_dataset/").toURI)
      .toString + "/"

    logger.info(s"Test resource found : $inputPath")

    val sparkSession2 = sparkSession

    import sparkSession2.implicits._

    val entries = sparkSession.read
      .parquet(inputPath + "matrixEntries")
      .as[MatrixEntry]
      .rdd

    // Matrix size is 9223 x 9223
    val matSize = 9223
    val blockSize = 200
    val partitionHeightInBlocks = 8
    val partitionWidthInBlocks = 4

    val oversample = 10
    val dim = 100
    val seed = 123
    val numPowerIterations = 1 // the image space of A is not properly captured with one power iterations,
    // so changing any parameters here or in the code should change the singular values

    val rsvdConf = RSVDConfig(
      embeddingDim = dim,
      powerIter = numPowerIterations,
      blockSize = blockSize,
      partitionWidthInBlocks = partitionWidthInBlocks,
      partitionHeightInBlocks = partitionHeightInBlocks,
      oversample = oversample,
      seed = seed,
      computeLeftSingularVectors = false,
      computeRightSingularVectors = false
    )

    val mat = BlockMatrix
      .fromMatrixEntries(entries,
                         matSize,
                         matSize,
                         rsvdConf.blockSize,
                         rsvdConf.partitionHeightInBlocks,
                         rsvdConf.partitionWidthInBlocks)
      .persist(StorageLevel.DISK_ONLY)

    val RsvdResults(_, singularValues, _) = RSVD.run(mat, rsvdConf, sc)

    logger.info(s"singular values : ${singularValues.toString()}")

    assert(singularValues(0) === 98.300 +- 0.001)
    assert(singularValues(1) === 84.311 +- 0.001)

    FileUtils.deleteDirectory(tmpDir)

  }

  test(
    "Test on real tiny dataset, with unconverged power iterations, rectangular case") {
    val tmpDir = Files.createTempDirectory("r-svd").toFile
    val basePath = Paths
      .get(Resources.getResource("tiny_dataset/").toURI)
      .toString + "/"

    logger.info(s"Test resource found : $basePath")

    // Matrix size is 9223 x 9223, but we will crop it to 9223 x 9023
    val matHeight = 9223
    val matWidth = 9023
    val blockSize = 200
    val partitionHeightInBlocks = 8
    val partitionWidthInBlocks = 4

    val oversample = 10
    val dim = 100
    val seed = 123
    val numPowerIterations = 1 // the image space of A is not properly captured with one power iterations,
    // so changing any parameters here or in the code should change the singular values

    val rsvdConf = RSVDConfig(
      embeddingDim = dim,
      powerIter = numPowerIterations,
      blockSize = blockSize,
      partitionWidthInBlocks = partitionWidthInBlocks,
      partitionHeightInBlocks = partitionHeightInBlocks,
      oversample = oversample,
      seed = seed,
      computeLeftSingularVectors = true,
      computeRightSingularVectors = false
    )

    val sparkSession2 = sparkSession

    import sparkSession2.implicits._

    val entries = sparkSession.read
      .parquet(basePath + "matrixEntries")
      .as[MatrixEntry]
      .rdd
      .filter(_.j < 9023)

    val mat = BlockMatrix
      .fromMatrixEntries(
        entries,
        matHeight = matHeight,
        matWidth = matWidth,
        rsvdConf.blockSize,
        partitionWidthInBlocks = rsvdConf.partitionWidthInBlocks,
        partitionHeightInBlocks = rsvdConf.partitionHeightInBlocks
      )
      .persist(StorageLevel.DISK_ONLY)

    val RsvdResults(_, singularValues, _) = RSVD.run(mat, rsvdConf, sc)

    logger.info(s"singular values : ${singularValues.toString()}")

    assert(singularValues(0) === 94.520 +- 0.001)
    assert(singularValues(1) === 84.205 +- 0.001)

    FileUtils.deleteDirectory(tmpDir)

  }
}
