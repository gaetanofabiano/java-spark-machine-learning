package it.fabiano.bigdata.spark.ml.es00;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.mllib.linalg.DenseVector;
/**
* Java-Spark-Training-Course
*
* @author  Gaetano Fabiano
* @version 1.0.0
* @since   2019-07-19 
*/
public class VectorsExamples {

	
	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName(VectorsExamples.class.getName()).setMaster("local[*]");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		double[] array = {34,100,185};
		int[] index = {0,1,2};
		
		DenseVector denseVector = new DenseVector(array);
		
		SparseVector sparseVector = new SparseVector(3, index, array);
		
		System.out.println(denseVector);
		System.out.println(sparseVector);
		
		
		sc.stop();
		sc.close();
	}
}