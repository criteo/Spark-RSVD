package com.criteo.rsvd

import java.nio.ByteBuffer

import com.esotericsoftware.kryo.Kryo
import com.typesafe.scalalogging.slf4j.StrictLogging
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
