import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{PCA,StandardScaler,VectorAssembler}

object PCA extends App{

  val spark = SparkSession.builder()
    .appName("PCA")
    .master("local[*]")
    .getOrCreate()

  Logger.getLogger("org")
    .setLevel(Level.ERROR)

  val path = "src/main/resources/PCA/Cancer_Data"
  val df = spark.read.option("inferSchema",true)
    .option("header",true)
    .csv(path)

  val assembler = new VectorAssembler()
    .setInputCols(df.columns)
    .setOutputCol("features")
    .transform(df)
    .select("features")

  val scaler  = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)
    .setWithMean(false)
    .fit(assembler)
    .transform(assembler)
    .select("scaledFeatures")

  val pca = new PCA().setK(4)
    .setInputCol("scaledFeatures")
    .setOutputCol("pcaFeatures")
    .fit(scaler)
    .transform(scaler)
    .show(truncate = false)

}
