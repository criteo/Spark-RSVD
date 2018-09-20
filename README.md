# Spark-RSVD
Spark-RSVD is a lib to compute approximate SVD decomposition of large sparse matrices (up to 100 million rows and columns) using an iterative algorithm for speed and efficiency.

Here is an example that will compute a 100-dimension embedding on a 200K * 200K matrix

```
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

val inputToDecompose = BlockMatrix.fromMatrixEntries(randomMatrixEntries,
                                           matHeight = matHeight,
                                           matWidth = matWidth,
                                           config.blockSize,
                                           config.partitionHeightInBlocks,
                                           config.partitionWidthInBlocks)

val RsvdResults(leftSingularVectors, singularValues, rightSingularVectors) =
  RSVD.run(inputToDecompose, config, sc)

//print the top 100 (embeddingDim=100) singular values in decreasing order:
println(singularValues.toString())

//fetch the left-singular vectors to driver, which will be a 100 Millions x 100 matrix.
//this is available because we set config.computeLeftSingularVectors = true
val leftSingularOnDriver = leftSingularVectors.get.toLocalMatrix

//fetch the right-singular vectors to driver, which will be a 100 Millions x 100 matrix.
//this is available because we set config.computeRightSingularVectors = true
val rightSingularOnDriver = rightSingularVectors.get.toLocalMatrix
```
