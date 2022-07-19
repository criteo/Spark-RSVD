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

import com.google.common.io.Resources
import com.typesafe.scalalogging.StrictLogging
import org.apache.commons.io.FileUtils
import org.scalactic.Tolerance._
import org.scalatest.FunSuite
import scala.collection.JavaConverters._

class RSVDWrapperTest
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

    val blockSize = 200
    val partitionHeightInBlocks = 8
    val partitionWidthInBlocks = 4

    val oversample = 10
    val dim = 100
    val seed = 123
    val numPowerIterations =
      1 // the image space of A is not properly captured with one power iterations,
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

    val (_, singularValuesArray, _) =
      RSVDWrapper.run(entries, "i", "j", "value", rsvdConf)

    val singularValuesList = singularValuesArray.asScala

    logger.info(
      s"singular values : ${singularValuesList.mkString("Array(", ", ", ")")}")

    assert(singularValuesList(0) === 95.297 +- 0.001)
    assert(singularValuesList(1) === 84.311 +- 0.001)

    FileUtils.deleteDirectory(tmpDir)

  }
}
