package it.fabiano.bigdata.spark.ml.examples;

// $example on$
import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
// $example off$

class JavaDecisionTreeRegression {

  public static void main(String[] args) {

    // $example on$
    SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTreeRegressionExample").setMaster("local[*]");;
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);

    // Load and parse the data file.
    String datapath = "data/mllib/sample_libsvm_data.txt";
    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
    // Split the data into training and test sets (30% held out for testing)
    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
    JavaRDD<LabeledPoint> trainingData = splits[0];
    JavaRDD<LabeledPoint> testData = splits[1];

    // Set parameters.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
    String impurity = "variance";
    int maxDepth = 5;
    int maxBins = 32;

    // Train a DecisionTree model.
    DecisionTreeModel model = DecisionTree.trainRegressor(trainingData,
      categoricalFeaturesInfo, impurity, maxDepth, maxBins);

    // Evaluate model on test instances and compute test error
    JavaPairRDD<Double, Double> predictionAndLabel =
      testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
    double testMSE = predictionAndLabel.mapToDouble(pl -> {
      double diff = pl._1() - pl._2();
      return diff * diff;
    }).mean();
    System.out.println("Test Mean Squared Error: " + testMSE);
    System.out.println("Learned regression tree model:\n" + model.toDebugString());

    // Save and load model
    model.save(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
    DecisionTreeModel sameModel = DecisionTreeModel
      .load(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
    // $example off$
  }
}
