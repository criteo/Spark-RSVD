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

import breeze.linalg.{DenseMatrix => BDM}
import org.scalatest.FunSuite

class CSRMatrixTest extends FunSuite {

  test("CSR matrix activeIterator should list the elements row-wise") {
    val builder = new CSRMatrix.Builder(3, 6)

    builder.add(0, 0, 1.0)
    builder.add(2, 0, 2.0)
    builder.add(0, 5, 3.0)
    builder.add(1, 4, 4.0)

    val CSRMat = builder.result

    val activeValues = List[((Int, Int), Double)](
      ((0, 0), 1.0),
      ((0, 5), 3.0),
      ((1, 4), 4.0),
      ((2, 0), 2.0)
    )
    assert(CSRMat.activeIterator.toList === activeValues)
  }

  test("CSR matrix transpose should work as expected") {
    val builder = new CSRMatrix.Builder(3, 6)

    builder.add(0, 0, 1.0)
    builder.add(2, 0, 2.0)
    builder.add(0, 5, 3.0)
    builder.add(1, 4, 4.0)

    val CSRMat = builder.result

    val localMatrix = BDM.zeros[Double](3, 6)
    CSRMat.activeIterator.foreach({ case ((i, j), v) => localMatrix(i, j) = v })

    val CSRMatTransposed = CSRMat.t

    val localMatrixTransposed = BDM.zeros[Double](6, 3)

    CSRMatTransposed.activeIterator.foreach({
      case ((i, j), v) => localMatrixTransposed(i, j) = v
    })

    assert(localMatrixTransposed === localMatrix.t)
  }

}
