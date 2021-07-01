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

import java.nio.ByteBuffer

import com.esotericsoftware.kryo.Kryo
import com.typesafe.scalalogging.StrictLogging
import de.javakaffee.kryoserializers.UnmodifiableCollectionsSerializer
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.{KryoRegistrator, KryoSerializer}
import org.apache.spark.{SparkConf, SparkContext}

import scala.reflect.ClassTag

object ReadingWritingData extends StrictLogging {

  def getInputDataSizeMB(inputPathPattern: String, sc: SparkContext): Int = {
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val path = new Path(inputPathPattern)
    (fs.globStatus(path).map(f => f.getLen).sum / 1024 / 1024).toInt
  }

  def loadMatrixEntries(inputPath: String,
                        singlePartitionSizeMB: Int,
                        sc: SparkContext): RDD[MatrixEntry] = {

    logger.info(s"Input matrix path: $inputPath")
    val inputDataSizeMB = getInputDataSizeMB(inputPath + "/*", sc)
    logger.info(s"Input size in MB: $inputDataSizeMB")
    val nbPartitions = Math.ceil(inputDataSizeMB / singlePartitionSizeMB).toInt
    logger.info(s"Number of partitions used for the input: $nbPartitions")
    makeRddFromKryoFile[MatrixEntry](sc, inputPath, Some(nbPartitions))
  }

  /** A modification of [[SparkContext.objectFile]] which expects Kryo. */
  def makeRddFromKryoFile[T: ClassTag](
      sc: SparkContext,
      path: String,
      minPartitionsOpt: Option[Int] = None): RDD[T] = {
    val minPartitions = minPartitionsOpt.getOrElse(sc.defaultMinPartitions)
    val serializer = new KryoSerializer(sc.getConf)
    sc.sequenceFile(path,
                    classOf[NullWritable],
                    classOf[BytesWritable],
                    minPartitions)
      .mapPartitions { it =>
        val instance = serializer.newInstance()
        it.flatMap {
          case (_, v) =>
            instance.deserialize[Array[T]](ByteBuffer.wrap(v.getBytes))
        }
      }
  }

  object RandomizedSVDKryoRegistrator extends KryoRegistrator {

    def registerClasses(kryo: Kryo): Unit = {
      UnmodifiableCollectionsSerializer.registerSerializers(kryo)
      kryo.register(classOf[MatrixEntry])
      kryo.register(classOf[Array[MatrixEntry]])
    }
  }

  def appendBasicRegistratorToSparkConf(sparkConf: SparkConf): SparkConf =
    appendRegistratorToSparkConf(sparkConf,
                                 RandomizedSVDKryoRegistrator.getClass.getName)

  def appendRegistratorToSparkConf(sparkConf: SparkConf,
                                   registratorName: String): SparkConf = {
    val oldValue = sparkConf.get("spark.kryo.registrator", "")
    if (oldValue == "") {
      sparkConf.set("spark.kryo.registrator", registratorName)
    } else {
      sparkConf.set("spark.kryo.registrator", oldValue + "," + registratorName)
    }
  }

}
