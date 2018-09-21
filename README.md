# Spark-RSVD
Spark-RSVD is a lib to compute approximate SVD decomposition of large sparse matrices (up to 100 million rows and columns) using an iterative algorithm for speed and efficiency.

The iterative algorithm is based on a random initial starting point, hence its name of Randomized SVD algorithm. It is described in \[[1](@references)\]. A tree reduce algorithm is used for fast QR decomposition of tall and skinny matrices \[[2](@references)\].

## Inputs and outputs

Spark-RSVD expects a sparse matrix of the type `BlockMatrix` as defined [here](src/main/scala/com/criteo/rsvd/BlockMatrix.scala#L250). To help creating such a matrix, a helper function (`FromMatrixEntries` [code](src/main/scala/com/criteo/rsvd/BlockMatrix.scala#L78)) is provided. It accepts an RDD of `MatrixEntry`, as defined by [MLLib](https://spark.apache.org/docs/2.1.0/api/java/org/apache/spark/mllib/linalg/distributed/MatrixEntry.html), which is a simple case class to indicate non-zero values in coordinate format. 

Spark-RSVD also needs some other parameters as inputs which are explained in the [parameters](@parameters) section below.

Spark-RSVD outputs the left singular vectors, the singular values and the right singular vectors (if needed). The singular values are stored as a [Breeze](https://github.com/scalanlp/breeze) `DenseVector[Double]`. The singular vectors are stored as a `SkinnyBlockMatrix` defined [here](src/main/scala/com/criteo/rsvd/SkinnyBlockMatrix.scala#L99).

## Parameters

Spark-RSVD needs some parameters which are collected together in a [case class](src/main/scala/com/criteo/rsvd/RSVD.scala#L24) called `RSVDConfig`. They are mostly self-explanatory, but here is a description of their meaning:

- `embeddingDim`: `Int`. Number of the approximate singular vectors and singular values to be computed
- `oversample`: `Int`. Number of surplus vectors which are needed to increase the precision of the singular vectors. See \[[1](@references)\] for more explanation.
- `powerIter`: `Int`. Number of iterations of the iterative algorithm. Only a couple are needed to get to a high precision (again, see \[[1](@references)\])
- `seed`: `Int`. Seed for the initialization of the random basis. Using always the same value should lead to repeatable results (though Spark may have unrepeatable results due to the varying order of execution of some operations)
- `blockSize`: `Int`. Size of the blocks used in `BlockMatrix` and `SkinnyBlockMatrix`. See the section [data format](@data-format) for more explanations.
- `partitionWidthInBlocks`: `Int`. Width of the partitions of the `BlockMatrix` in number of blocks. The `SkinnyBlockMatrix` is also partitioned vertically with the same number of blocks for consistency during the matrix-vector multiplication. See the section [data format](@data-format) for more explanations.
- `partitionWidthInBlocks`: `Int`. Height of the partitions of the `BlockMatrix` in number of blocks. See the section [data format](@data-format) for more explanations.
- `computeLeftSingularVectors` and `computeRightSingularVectors`: `Boolean`. Indicates whether the left singular vectors and the right singular vectors should be computed.

### Sensible configuration

## Data format

## Example

Here is an example that will compute a 100-dimension embedding on a 200K * 200K matrix

```Scala
import com.criteo.rsvd._

// create spark context
val sc: SparkContext = new SparkContext(...)

// create RSVD configuration
val config = RSVDConfig(
  embeddingDim = 100,
  oversample = 30,
  powerIter = 1,
  seed = 0,
  blockSize = 50000,
  partitionWidthInBlocks = 35,
  partitionHeightInBlocks = 10,
  computeLeftSingularVectors = true,
  computeRightSingularVectors = true
)

val matHeight = 200000 // 200K
val matWidth = 200000 // 200K
val numNonZeroEntries = 400000 // 400K

//generate a sparse random matrix as an input (doesn't have to be symmetric)
val randomMatrixEntries = sc.parallelize(0 until numNonZeroEntries).map {
  idx =>
    val random = new Random(42 + idx)
    MatrixEntry(random.nextInt(matHeight), //row index
                random.nextInt(matWidth), //column index
                random.nextGaussian()) //entry value
}

val matrixToDecompose = BlockMatrix.fromMatrixEntries(randomMatrixEntries,
                                           matHeight = matHeight,
                                           matWidth = matWidth,
                                           config.blockSize,
                                           config.partitionHeightInBlocks,
                                           config.partitionWidthInBlocks)

val RsvdResults(leftSingularVectors, singularValues, rightSingularVectors) =
  RSVD.run(matrixToDecompose, config, sc)

//print the top 100 (embeddingDim=100) singular values in decreasing order:
println(singularValues.toString())

//fetch the left-singular vectors to driver, which will be a 200K x 100 matrix.
//this is available because we set config.computeLeftSingularVectors = true
val leftSingularOnDriver = leftSingularVectors.get.toLocalMatrix

//fetch the right-singular vectors to driver, which will be a 200K x 100 matrix.
//this is available because we set config.computeRightSingularVectors = true
val rightSingularOnDriver = rightSingularVectors.get.toLocalMatrix
```

## References:

\[1\] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.

\[2\] Constantine, P. G., & Gleich, D. F. (2011, June). Tall and skinny QR factorizations in MapReduce architectures. In Proceedings of the second international workshop on MapReduce and its applications (pp. 43-50). ACM.
