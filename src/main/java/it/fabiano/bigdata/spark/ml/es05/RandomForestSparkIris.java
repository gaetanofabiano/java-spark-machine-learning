package it.fabiano.bigdata.spark.ml.es05;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

/**
 * Apache Spark MLLib Java algorithm for classifying the Iris Species
 * into three categories using a Random Forest Classification algorithm.
 */


/**
* Java-Spark-Machine-Learning-Course
*
* @author  Gaetano Fabiano
* @version 1.1.0
* @since   2019-07-19 
* @updated 2020-07-01 
*/
public class RandomForestSparkIris {

   

    public static void main(String[] args) {

        
        SparkSession sparkSession = SparkSession.builder().appName("SparkIris").master("local[*]").getOrCreate();
        
        

        // load dataset, which has a header at the first row
        Dataset<Row> rawData = sparkSession.read().option("header", "true").csv("in/iris.csv");

        // cast the values of the features to doubles for usage in the feature column vector
        Dataset<Row> transformedDataSet = rawData.withColumn("SepalLengthCm", rawData.col("SepalLengthCm").cast("double"))
                .withColumn("SepalWidthCm", rawData.col("SepalWidthCm").cast("double"))
                .withColumn("PetalLengthCm", rawData.col("PetalLengthCm").cast("double"))
                .withColumn("PetalWidthCm", rawData.col("PetalWidthCm").cast("double"));

        // add a numerical label column for the Random Forest Classifier
        transformedDataSet = transformedDataSet
                .withColumn("label", when(col("Species").equalTo("Iris-setosa"),1)
                .when(col("Species").equalTo("Iris-versicolor"),2)
                .otherwise(3));

        // identify the feature colunms
        String[] inputColumns = {"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"};
        VectorAssembler assembler = new VectorAssembler().setInputCols(inputColumns).setOutputCol("features");
        Dataset<Row> featureSet = assembler.transform(transformedDataSet);

        // split data random in trainingset (70%) and testset (30%) using a seed so results can be reproduced
        long seed = 5043;
        Dataset<Row>[] trainingAndTestSet = featureSet.randomSplit(new double[]{0.7, 0.3}, seed);
        Dataset<Row> trainingSet = trainingAndTestSet[0];
        Dataset<Row> testSet = trainingAndTestSet[1];

        trainingSet.show();

        // train the algorithm based on a Random Forest Classification Algorithm with default values
        RandomForestClassifier randomForestClassifier = new RandomForestClassifier().setSeed(seed);
        RandomForestClassificationModel model = randomForestClassifier.fit(trainingSet);

        // test the model against the testset and show results
        Dataset<Row> predictions = model.transform(testSet);
        predictions.select("id", "label", "prediction").show(5);

        // evaluate the model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        System.out.println("accuracy: " + evaluator.evaluate(predictions));
    }
}
