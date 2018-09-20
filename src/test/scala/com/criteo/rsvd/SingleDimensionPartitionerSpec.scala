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

import org.scalatest.FunSuite
import org.scalatest.prop.TableDrivenPropertyChecks

class SingleDimensionPartitionerSpec
    extends FunSuite
    with TableDrivenPropertyChecks
    with PerTestSparkSession {
  test(
    "Partitioner should partition square and skinny matrices with the same " +
      "number of columns / rows per partition") {

    val numRowBlocks = 5
    val numBlocksPerPartition = 2

    val indices = Table(
      ("SkinnyBlockMatrixIndex", "BlockMatrixIndex", "ExpectedPartitionId"),
      (0, (0, 0), 0),
      (0, (1, 0), 0),
      (0, (0, 1), 0),
      (1, (4, 0), 0),
      (2, (3, 2), 1)
    )

    val partitioner =
      SingleDimensionPartitioner(numRowBlocks, numBlocksPerPartition)
    forAll(indices) {
      (skinnyIndex: Int,
       squareIndex: (Int, Int),
       expectedPartitionIndex: Int) =>
        assert(
          partitioner.getPartition(skinnyIndex) === partitioner.getPartition(
            squareIndex))
        assert(partitioner.getPartition(skinnyIndex) === expectedPartitionIndex)
    }
  }

  test("createCompatibleIndicesRDD works") {
    val numRowBlocks = 5
    val numBlocksPerPartition = 2
    val partitioner =
      SingleDimensionPartitioner(numRowBlocks, numBlocksPerPartition)

    val rdd = partitioner.createCompatibleIndicesRDD(sc)

    assert(rdd.partitions.length == 3)

    val data = rdd
      .mapPartitionsWithIndex {
        case (idx, it) => Iterator((idx, it.map(_._1).toList))
      }
      .collect()
      .sortBy(_._1)

    assert(
      data ===
        Array(
          (0, List(0, 1)),
          (1, List(2, 3)),
          (2, List(4))
        )
    )

  }
}
