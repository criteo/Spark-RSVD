package com.criteo.rsvd

import breeze.linalg.DenseMatrix
import org.scalatest.FunSuite

class UtilsTest extends FunSuite with PerTestSparkSession {

  test("isAPartition should work") {

    val incorrect1 = Array(Set(2, 3), Set(3, 4, 5))
    val incorrect2 = Array(Set(2), Set(4, 5))
    val incorrect3 = Array(Set(2, 3, 4, 5), Set.empty[Int])
    val incorrect4 = Array.empty[Set[Int]]

    assert(!Utils.isAPartition(incorrect1, 2, 5))
    assert(!Utils.isAPartition(incorrect2, 2, 5))
    assert(!Utils.isAPartition(incorrect3, 2, 5))
    assert(!Utils.isAPartition(incorrect4, 2, 5))

    val correct1 = Array(Set(2), Set(3), Set(4), Set(5))
    val correct2 = Array(Set(2), Set(3, 4), Set(5))
    val correct3 = Array(Set(2, 3, 4, 5))

    assert(Utils.isAPartition(correct1, 2, 5))
    assert(Utils.isAPartition(correct2, 2, 5))
    assert(Utils.isAPartition(correct3, 2, 5))

    //incorrect bounds:
    assert(!Utils.isAPartition(correct1, 2, 6))
    assert(!Utils.isAPartition(correct1, 1, 5))
    assert(!Utils.isAPartition(correct1, 3, 5))
    assert(!Utils.isAPartition(correct1, 2, 4))
  }

  test("IterRows should work") {
    val matrix = DenseMatrix.zeros[Double](2, 3)

    matrix(0, 0) = 1.0
    matrix(0, 1) = 2.0
    matrix(0, 2) = 3.0

    matrix(1, 0) = 10.0
    matrix(1, 1) = 20.0
    matrix(1, 2) = 30.0

    val it = Utils.rowIter(matrix)

    val firstLine = it.next()
    val secondLine = it.next()
    assert(!it.hasNext)
    assert(firstLine.data === Array(1.0, 2.0, 3.0))
    assert(secondLine.data === Array(10.0, 20.0, 30.0))
  }

}
