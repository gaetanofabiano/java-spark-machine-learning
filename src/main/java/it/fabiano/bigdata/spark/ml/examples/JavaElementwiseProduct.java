package it.fabiano.bigdata.spark.ml.examples;

// $example on$
import java.util.Arrays;
// $example off$

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
// $example on$
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.feature.ElementwiseProduct;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
// $example off$

public class JavaElementwiseProduct {
  public static void main(String[] args) {

    SparkConf conf = new SparkConf().setAppName("JavaElementwiseProductExample").setMaster("local[*]");;
    JavaSparkContext jsc = new JavaSparkContext(conf);

    // $example on$
    // Create some vector data; also works for sparse vectors
    JavaRDD<Vector> data = jsc.parallelize(Arrays.asList(
      Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(4.0, 5.0, 6.0)));
    Vector transformingVector = Vectors.dense(0.0, 1.0, 2.0);
    ElementwiseProduct transformer = new ElementwiseProduct(transformingVector);

    // Batch transform and per-row transform give the same results:
    JavaRDD<Vector> transformedData = transformer.transform(data);
    JavaRDD<Vector> transformedData2 = data.map(transformer::transform);
    // $example off$

    System.out.println("transformedData: ");
    transformedData.foreach(System.out::println);

    System.out.println("transformedData2: ");
    transformedData2.foreach(System.out::println);

    jsc.stop();
  }
}
