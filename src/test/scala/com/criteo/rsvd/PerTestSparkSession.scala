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

import java.io.File
import java.nio.file.{Files, Path}
import java.util.concurrent.locks.ReentrantLock

import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.scalatest.{BeforeAndAfterEach, Suite}

import scala.reflect.ClassTag
import scala.util.control.NonFatal

object LocalSparkSession {
  private[this] val lock = new ReentrantLock()

  def acquire(): Unit = lock.lock()

  def release(): Unit = lock.unlock()

  def builder: SparkSession.Builder = {
    SparkSession
      .builder()
      .master("local[*]")
      .appName("test")
      .config("spark.ui.enabled", false)
  }
}

/**
  * Runs each test with a new [[SparkSession]] instance.
  *
  * Due to the fact that [[SparkContext]] is JVM-global, the tests are
  * forced to execute sequentially to ensure isolation.
  */
trait PerTestSparkSession extends BeforeAndAfterEach { self: Suite =>

  protected[this] var currentSession: Option[SparkSession] = None
  protected[this] var checkpointDir: Option[Path] = None

  /** A common alias to [[sparkContext]]. */
  final def sc: SparkContext = sparkContext

  final def sparkSession: SparkSession = getOrCreateSession
  final def sparkContext: SparkContext = sparkSession.sparkContext
  final def sqlContext: SQLContext = sparkSession.sqlContext

  /**
    * Properties to be added to [[org.apache.spark.SparkConf]] before
    * running the test.
    */
  def sparkConf: Map[String, Any] = Map()

  def toRDD[T: ClassTag](input: Seq[T]): RDD[T] = sc.parallelize(input)

  def toArray[T](input: RDD[T]): Array[T] = input.collect()

  protected def closeSession() = {
    currentSession.foreach(_.stop())
    currentSession = None
    try {
      checkpointDir.foreach(path =>
        FileUtils.deleteDirectory(new File(path.toString)))
    } catch {
      case NonFatal(_) =>
    }
    checkpointDir = None
    LocalSparkSession.release()
  }

  private def getOrCreateSession = synchronized {
    if (currentSession.isEmpty) {
      val builder = LocalSparkSession.builder
      for ((key, value) <- sparkConf) {
        builder.config(key, value.toString)
      }
      currentSession = Some(builder.getOrCreate())
      checkpointDir =
        Some(Files.createTempDirectory("spark-unit-test-checkpoint-"))
      currentSession.get.sparkContext
        .setCheckpointDir(checkpointDir.get.toString)
        currentSession.get.sparkContext.setLogLevel("WARN")
    }
    currentSession.get
  }

  override def beforeEach(): Unit = {
    LocalSparkSession.acquire()
    super.beforeEach()
  }

  override def afterEach(): Unit = {
    try {
      super.afterEach()
    } finally {
      closeSession()
    }
  }
}
