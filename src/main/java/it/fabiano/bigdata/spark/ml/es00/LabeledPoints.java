package it.fabiano.bigdata.spark.ml.es00;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
/**
* Java-Spark-Machine-Learning-Course
*
* @author  Gaetano Fabiano
* @version 1.1.0
* @since   2019-07-19 
* @updated 2020-07-01 
*/


public class LabeledPoints {


	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName(LabeledPoints.class.getName()).setMaster("local[*]");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		double[] array = {34,100,185};


		DenseVector denseVector = new DenseVector(array);


		LabeledPoint labeledPoint1 = new LabeledPoint(1.0, denseVector);
		LabeledPoint labeledPoint2 = new LabeledPoint(0.0, new DenseVector(new double[]{33,101,185}));
		LabeledPoint labeledPoint3 = new LabeledPoint(1.0, new DenseVector(new double[]{31,59,177}));


		System.out.println(labeledPoint1);
		System.out.println(labeledPoint2);
		System.out.println(labeledPoint3);

		sc.close();
	}
}