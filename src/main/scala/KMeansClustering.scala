import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans

object ClusteringKMeans extends App{

  val spark = SparkSession.builder().appName("kmeansClustering")
    .master("local[*]")
    .getOrCreate()

  Logger.getLogger("org").setLevel(Level.ERROR)

  val path = "src/main/resources/Clustering/Wholesale customers data.csv"
  val df = spark.read.option("inferSchema",true)
    .option("header",true)
    .csv(path)

  val feature_data = df.select("Fresh", "Milk","Grocery","Frozen","Detergents_Paper","Delicassen")

  val assembler = new VectorAssembler()
    .setInputCols(feature_data.columns)
    .setOutputCol("features")

  val training_data = assembler.transform(df)
    .select("features")

  val kMeans = new KMeans()
    .setK(2)

  val model = kMeans.fit(training_data)
  val modelSummary = model.summary


  println(s"training cost - ${modelSummary.trainingCost}")
  
}
