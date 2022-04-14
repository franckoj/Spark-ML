import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, hour}

object LogisticRegression extends App{
  val spark = SparkSession.builder().appName("LinearRegression")
    .master("local[*]")
    .getOrCreate()
  Logger.getLogger("Logistic").setLevel(Level.ERROR)

  val path = "src/main/resources/Classification/advertising.csv"

  val df = spark.read.option("header",true)
    .option("inferSchema",true)
    .csv(path)

  val ldata = df.withColumnRenamed("Clicked on Ad", "label")
    .withColumn("Timestamp", hour(col("Timestamp")))
    .drop("City","Country","Ad Topic Line")

  ldata.printSchema()

  import org.apache.spark.ml.classification.LogisticRegression

  val assembler = new VectorAssembler()
    .setInputCols(ldata.columns)
    .setOutputCol("features")
//    .transform(ldata)
//    .select("label","features")



  val Array(train_data, test_data) = ldata.randomSplit(Array(0.7,0.3),seed = 12345)

  val lcModel = new LogisticRegression()
//    .fit(assembler)

  val pipeline = new Pipeline()
    .setStages(Array(assembler, lcModel))
    .fit(train_data)
    .transform(test_data)

  import spark.implicits._
  val results = pipeline
    .select("prediction","label").as[(Double,Double)].rdd


  val metrics: Matrix = new MulticlassMetrics(results)
    .confusionMatrix

  println(s"ConfusionMatrix - $metrics")
}
