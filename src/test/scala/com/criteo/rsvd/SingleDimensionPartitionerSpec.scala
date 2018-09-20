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
