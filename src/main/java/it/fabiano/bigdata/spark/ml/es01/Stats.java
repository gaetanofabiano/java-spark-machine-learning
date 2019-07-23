package it.fabiano.bigdata.spark.ml.es01;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
/**
* Java-Spark-Training-Course
*
* @author  Gaetano Fabiano
* @version 1.0.0
* @since   2019-07-19 
*/
public class Stats {


	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setAppName(Stats.class.getName()).setMaster("local[*]");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		List<Vector> list = new ArrayList<Vector>();
		
		
		list.add(new DenseVector(new double[]{1,101,185}));
		list.add(new DenseVector(new double[]{2,102,185}));
		list.add(new DenseVector(new double[]{3,101,185}));
		list.add(new DenseVector(new double[]{4,102,185}));
		list.add(new DenseVector(new double[]{5,101,185}));
		list.add(new DenseVector(new double[]{6,102,185}));
		list.add(new DenseVector(new double[]{7,101,185}));
		list.add(new DenseVector(new double[]{8,102,185}));
		list.add(new DenseVector(new double[]{9,101,185}));
		list.add(new DenseVector(new double[]{10,102,185}));
		list.add(new DenseVector(new double[]{11,101,185}));
		list.add(new DenseVector(new double[]{12,102,185}));
		list.add(new DenseVector(new double[]{13,101,432}));
		list.add(new DenseVector(new double[]{14,342,342}));
		list.add(new DenseVector(new double[]{15,3432,185}));
		list.add(new DenseVector(new double[]{16,43,342}));
		list.add(new DenseVector(new double[]{17,43,342}));
		list.add(new DenseVector(new double[]{18,43,54}));
		list.add(new DenseVector(new double[]{19,21,6}));
		list.add(new DenseVector(new double[]{20,99,5}));
		
		
		JavaRDD<Vector> rdds = sc.parallelize(list);
		
		MultivariateStatisticalSummary summary = Statistics.colStats(rdds.rdd());
		
		System.out.println("mean: "+summary.mean());
		System.out.println("variance: "+summary.variance());
		System.out.println("numNonZeros: "+summary.numNonzeros());
		System.out.println("count: "+summary.count());
		System.out.println("max: "+summary.max());
		System.out.println("min: "+summary.min());

		

		sc.close();
	}
}