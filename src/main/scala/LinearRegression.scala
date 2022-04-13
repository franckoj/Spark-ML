import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler

object LinearRegression extends App{
  val spark = SparkSession.builder().appName("LinearRegression")
    .master("local[*]")
    .getOrCreate()

  Logger.getLogger("org").setLevel(Level.ERROR)


  def clean_USA_housing()={

    val path = "src/main/resources/Clean_USA_Housing.csv"
    val data = spark.read.format("csv")
      .option("inferSchema",true)
      .option("header",true)
      .load(path)

    val df = data.withColumnRenamed("Price","label")

    val array = df.columns

    //converting all columns into an array of rows
    val assembler = new VectorAssembler()
      .setInputCols(array)
      .setOutputCol("features")

    val output = assembler.transform(df)
      .select("label","features")

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val lrModel = lr.fit(output)
    val trainingSummary = lrModel.summary
    trainingSummary.predictions.show()
    trainingSummary.r2

  }


  def lr_documentation_set(): Unit ={
    val path = "src/main/resources/sample_linear_regression_data.txt"
    // Load training data
    val training = spark.read.format("libsvm")
      .load(path)

    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  }
  clean_USA_housing()

}
