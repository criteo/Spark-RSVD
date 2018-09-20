# Spark-RSVD
Spark-RSVD is a lib to compute approximate SVD decomposition of large sparse matrices (up to 100 million rows and columns) using an iterative algorithm for speed and efficiency.

The iterative algorithm is based on a random initial starting point, hence its name of Randomized SVD algorithm. It is described in \[ [1](@references) \]. A tree reduce algorithm is used for fast QR decomposition of tall and skinny matrices \[ [2](@references) \].

Here is an example that will compute a 100-dimension embedding on a 200K * 200K matrix

```
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

\[[1]\] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.
\[[2]\] Constantine, P. G., & Gleich, D. F. (2011, June). Tall and skinny QR factorizations in MapReduce architectures. In Proceedings of the second international workshop on MapReduce and its applications (pp. 43-50). ACM.
