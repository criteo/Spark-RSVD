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

import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.{functions => sf}

import scala.jdk.CollectionConverters._

object RSVDWrapper extends StrictLogging {

  lazy val spark: SparkSession = {
    SparkSession.builder().getOrCreate()
  }

  /** Launch the R-SVD decomposition of a given matrix.
    * @param inputMatrix the dataframe to run RSVD on. It is expected to have three columns, one for the row indices
    *                    one for the column indices and one for the matrix entries value.
    * @param rowIndexColumn the name of the dataframe column holding the row indices
    * @param columnIndexColumn the name of the dataframe column holding the column indices
    * @param valueIndexColumn  the name of the dataframe column holding the matrix entries value
    * @param config RSVD configuration parameters
    * @return A tuple with
    *
    *         * A DataFrame containing the left singular vectors and the row index they correspond to
    *           (or none if computeLeftSingularVectors is false)
    *
    *         * An array containing the singular values
    *
    *         * A DataFrame containing the right singular vectors and the column index they correspond to
    *           (or none if computeRightSingularVectors is false)
    */
  def run(
      inputMatrix: DataFrame,
      rowIndexColumn: String,
      columnIndexColumn: String,
      valueIndexColumn: String,
      config: RSVDConfig)
      : (Option[DataFrame], java.util.List[Double], Option[DataFrame]) = {
    import spark.implicits._

    val matrixEntries = inputMatrix
      .select(
        sf.col(rowIndexColumn).alias("i"),
        sf.col(columnIndexColumn).alias("j"),
        sf.col(valueIndexColumn).alias("value")
      )

    val matrixStats = matrixEntries.agg(sf.max($"i"), sf.max($"j")).head()
    // We add one because the indices are 0 indexed
    val (maxHeight, maxWidth) =
      (matrixStats.getLong(0) + 1, matrixStats.getLong(1) + 1)

    val blockMatrix = BlockMatrix.fromMatrixEntries(
      matrixEntries.as[MatrixEntry].rdd,
      matHeight = maxHeight,
      matWidth = maxWidth,
      config.blockSize,
      config.partitionHeightInBlocks,
      config.partitionWidthInBlocks)

    val RsvdResults(leftSingularVectors, singularValues, rightSingularVectors) =
      RSVD.run(blockMatrix, config, spark.sparkContext)

    def processSingularVectors(
        singularVectors: Option[SkinnyBlockMatrix],
        indexColumn: String): Option[DataFrame] = {
      singularVectors
        .map(vectors => vectors.toIndexedEmbeddings)
        .map(_.map { case (i, embedding) =>
          (i, embedding.data)
        }.toDF(indexColumn, "embedding"))
    }

    val leftSingularVectorsDF =
      processSingularVectors(leftSingularVectors, rowIndexColumn)
    val rightSingularVectorsDF =
      processSingularVectors(rightSingularVectors, rowIndexColumn)
    val singularValuesJavaList = singularValues.data.toList.asJava

    (leftSingularVectorsDF, singularValuesJavaList, rightSingularVectorsDF)
  }

}
