package it.fabiano.bigdata.spark.ml.examples;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

// $example on$
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
// $example off$

public class JavaGaussianMixture {
  public static void main(String[] args) {

    SparkConf conf = new SparkConf().setAppName("JavaGaussianMixtureExample").setMaster("local[*]");;
    JavaSparkContext jsc = new JavaSparkContext(conf);

    // $example on$
    // Load and parse data
    String path = "data/mllib/gmm_data.txt";
    JavaRDD<String> data = jsc.textFile(path);
    JavaRDD<Vector> parsedData = data.map(s -> {
      String[] sarray = s.trim().split(" ");
      double[] values = new double[sarray.length];
      for (int i = 0; i < sarray.length; i++) {
        values[i] = Double.parseDouble(sarray[i]);
      }
      return Vectors.dense(values);
    });
    parsedData.cache();

    // Cluster the data into two classes using GaussianMixture
    GaussianMixtureModel gmm = new GaussianMixture().setK(2).run(parsedData.rdd());

    // Save and load GaussianMixtureModel
    gmm.save(jsc.sc(), "target/org/apache/spark/JavaGaussianMixtureExample/GaussianMixtureModel");
    GaussianMixtureModel sameModel = GaussianMixtureModel.load(jsc.sc(),
      "target/org.apache.spark.JavaGaussianMixtureExample/GaussianMixtureModel");

    // Output the parameters of the mixture model
    for (int j = 0; j < gmm.k(); j++) {
      System.out.printf("weight=%f\nmu=%s\nsigma=\n%s\n",
        gmm.weights()[j], gmm.gaussians()[j].mu(), gmm.gaussians()[j].sigma());
    }
    // $example off$

    jsc.stop();
  }
}
