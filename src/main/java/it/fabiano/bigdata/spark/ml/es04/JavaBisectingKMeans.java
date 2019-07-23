package it.fabiano.bigdata.spark.ml.es04;


import java.util.Arrays;
import java.util.List;
// $example off$
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
// $example on$
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.BisectingKMeans;
import org.apache.spark.mllib.clustering.BisectingKMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
* Java-Spark-Training-Course
*
* @author  Gaetano Fabiano
* @version 1.0.0
* @since   2019-07-19 
*/
public class JavaBisectingKMeans {
  public static void main(String[] args) {
    SparkConf sparkConf = new SparkConf().setAppName("JavaBisectingKMeans").setMaster("local[*]");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);

   
    List<Vector> localData = Arrays.asList(
      Vectors.dense(0.1, 0.1),   Vectors.dense(0.3, 0.3),
      Vectors.dense(10.1, 10.1), Vectors.dense(10.3, 10.3),
      Vectors.dense(20.1, 20.1), Vectors.dense(20.3, 20.3),
      Vectors.dense(30.1, 30.1), Vectors.dense(30.3, 30.3)
    );
    JavaRDD<Vector> data = sc.parallelize(localData, 2);

    BisectingKMeans bkm = new BisectingKMeans()
      .setK(4);
    BisectingKMeansModel model = bkm.run(data);

    System.out.println("Compute Cost: " + model.computeCost(data));

    Vector[] clusterCenters = model.clusterCenters();
    for (int i = 0; i < clusterCenters.length; i++) {
      Vector clusterCenter = clusterCenters[i];
      System.out.println("Cluster Center " + i + ": " + clusterCenter);
    }
   

    sc.stop();
    sc.close();
  }
}
