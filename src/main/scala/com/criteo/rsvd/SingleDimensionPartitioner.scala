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

import com.google.common.math.LongMath
import com.google.common.primitives.Ints
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partitioner, SparkContext}

/**
  * SingleDimensionPartitioner is designed for BlockMatrix (A) times
  * SkinnyBlockMatrix (B) multiplication. It requires that columns of A are
  * partitioned in exactly the same way as rows of B. And that those
  * instances of the partitioner are "equal" when compared.
  *            |* * * * * *|
  *  Let A =   |* * * * * *|
  *            |* * * * * *|
  *            |* * * * * *|
  *            |* * * * * *|
  *            |* * * * * *|
  *
  *  The SingleDimensionPartitioner should partition it by columns:
  *            |* *|* *|* *|
  *            |* *|* *|* *|
  *            |* *|* *|* *|
  *            |* *|* *|* *|
  *            |* *|* *|* *|
  *            |* *|* *|* *|
  *
  *           | * |
  *  Let B =  | * |
  *           | * |
  *           | * |
  *           | * |
  *           | * |
  *
  *  Then the SingleDimensionPartitioner should partition it by rows of the same
  *  width as column partition of A:
  *           | * |
  *           |_*_|
  *           | * |
  *           |_*_|
  *           | * |
  *           | * |
  */
case class SingleDimensionPartitioner(dimensionSize: Long,
                                      dimensionSizePerPartition: Int)
    extends Partitioner {

  require(
    dimensionSize >= 1,
    s"The dimension size has to be a positive integer and it is $dimensionSize")

  require(
    dimensionSizePerPartition >= 1,
    s"The dimension size per partition has to be a positive integer and it is $dimensionSizePerPartition")

  override def numPartitions: Int =
    Ints.checkedCast(
      LongMath
        .divide(dimensionSize, dimensionSizePerPartition, RoundingMode.CEILING))

  override def getPartition(key: Any): Int = {
    key match {
      case rowIndex: Int =>
        getPartitionId(rowIndex) // SkinnyBlockMatrix
      case (_: Int, colIndex: Int) =>
        getPartitionId(colIndex) // BlockMatrix
      case _ =>
        throw new IllegalArgumentException(s"Unrecognized key: $key.")
    }
  }

  private def getPartitionId(j: Int): Int = {
    require(0 <= j && j < dimensionSize,
            s"Dimension index $j out of range [0, $dimensionSize).")
    j / dimensionSizePerPartition
  }

  /**
    * Create a RDD of integers from 0 to dimensionSize-1, with a partitioning
    * compatible with this partitioner. For now we are calling partitionBy but
    * it could be possible to create a compatible RDD without creating a
    * Shuffle Dependency
    */
  def createCompatibleIndicesRDD(sc: SparkContext): RDD[(Int, Int)] = {
    sc.parallelize(0 until dimensionSize.toInt, numPartitions)
      .map(idx => idx -> idx)
      .partitionBy(this)
  }
}
