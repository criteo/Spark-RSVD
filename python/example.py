import getpass
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import pyspark
import scipy.sparse as sps

JAR_URL_FORMAT = (
    "https://repository.sonatype.org/service/local/repositories/central-proxy"
    "/content/com/criteo/rsvd/{rsvd_jar_version}/rsvd-{rsvd_jar_version}.jar"
)


def download_rsvd_jar(
    rsvd_jar_version: str = "1.1-spark3",
) -> str:
    local_jar_path = os.path.join(
        tempfile.gettempdir(),
        getpass.getuser(),
        "com",
        "criteo",
        "rsvd",
        rsvd_jar_version,
        f"rsvd-{rsvd_jar_version}-uber.jar",
    )
    if os.path.exists(local_jar_path):
        return local_jar_path

    local_jar_dir = os.path.dirname(local_jar_path)
    os.makedirs(local_jar_dir, exist_ok=True)

    jar_url = JAR_URL_FORMAT.format(rsvd_jar_version=rsvd_jar_version)
    urllib.request.urlretrieve(jar_url, local_jar_path)
    return local_jar_path


def jvm_from_sparkcontext():
    if pyspark.SparkContext._jvm is None:
        raise Exception("Java gateway is not launched. Create a pyspark.SparkContext first.")
    return pyspark.SparkContext._jvm


@dataclass
class RSVDConfig:
    embedding_dim: int = 100
    oversample: int = 30
    power_iter: int = 1
    seed: int = 0
    block_size: int = 50000
    partition_width_in_blocks: int = 35
    partition_height_in_blocks: int = 10
    compute_left_singular_vectors: bool = True
    compute_right_singular_vectors: bool = False

    @property
    def jvm_config(self):
        jvm = jvm_from_sparkcontext()
        return jvm.com.criteo.rsvd.RSVDConfig(
            self.embedding_dim,
            self.oversample,
            self.power_iter,
            self.seed,
            self.block_size,
            self.partition_width_in_blocks,
            self.partition_height_in_blocks,
            self.compute_left_singular_vectors,
            self.compute_right_singular_vectors,
        )


def run_rvsd(
    input_matrix_df: pyspark.sql.DataFrame,
    row_index_column: str,
    column_index_column: str,
    value_column: str,
    config: RSVDConfig,
) -> Tuple[Optional[pyspark.sql.DataFrame], List[float], Optional[pyspark.sql.DataFrame]]:
    jvm = jvm_from_sparkcontext()
    result = jvm.com.criteo.rsvd.RSVDWrapper.run(
        input_matrix_df._jdf,
        row_index_column,
        column_index_column,
        value_column,
        config.jvm_config,
    )
    left_embeddings_df = pyspark.sql.DataFrame(result._1().get(), ss._wrapped) if not result._1().isEmpty() else None
    singular_values_array = result._2()
    right_embeddings_df = pyspark.sql.DataFrame(result._3().get(), ss._wrapped) if not result._3().isEmpty() else None
    return left_embeddings_df, singular_values_array, right_embeddings_df


if __name__ == "__main__":
    jar_path = download_rsvd_jar()

    ss = (
        pyspark.sql.SparkSession.builder.master("local")
        .appName("RSVD example")
        .config("spark.jars", jar_path)
        .getOrCreate()
    )

    config = RSVDConfig(
        embedding_dim=100,
        oversample=30,
        power_iter=1,
        seed=0,
        block_size=1_000,
        partition_width_in_blocks=35,
        partition_height_in_blocks=10,
        compute_left_singular_vectors=True,
        compute_right_singular_vectors=True,
    )

    mat_height = 10_000
    mat_width = 10_000
    num_non_zero_entries = 20_000
    density = num_non_zero_entries / (mat_height * mat_width)

    random_matrix = sps.random(mat_height, mat_width, density)
    random_matrix_pdf = pd.DataFrame({"row": random_matrix.row, "col": random_matrix.col, "data": random_matrix.data})
    random_matrix_df = ss.createDataFrame(random_matrix_pdf)

    left_embeddings_df, singular_values_array, right_embeddings_df = run_rvsd(
        random_matrix_df, row_index_column="row", column_index_column="col", value_column="data", config=config
    )

    print(singular_values_array)
    left_embeddings_df.show()
    right_embeddings_df.show()
