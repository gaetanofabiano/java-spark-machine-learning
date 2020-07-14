package it.fabiano.bigdata.spark.ml.es00;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

/**
* Java-Spark-Machine-Learning-Course
*
* @author  Gaetano Fabiano
* @version 1.1.0
* @since   2019-07-19 
* @updated 2020-07-01 
*/

public class Matrices {


	public static void main(String[] args) {
		
		SparkConf sparkConf = new SparkConf().setAppName(Matrices.class.getName()).setMaster("local[*]");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);


		
		List<Vector> list = new ArrayList<Vector>();
	
	
		list.add(new DenseVector(new double[]{33,101,185}));
		list.add(new DenseVector(new double[]{34,102,185}));
		
		JavaRDD<Vector> rows = sc.parallelize(list);
		 
		// Create a RowMatrix from JavaRDD<Vector>.
	    RowMatrix matrix1 = new RowMatrix(rows.rdd());
	    
	    
	    System.out.println("num of Cols: "+matrix1.numCols());
	    System.out.println("num of Rows: "+matrix1.numRows());
	    
	    
	    //https://en.wikipedia.org/wiki/Covariance_matrix
	   
	    Matrix covariance = matrix1.computeCovariance();
	    System.out.println("covariance Cols: "+covariance.numCols());
	    System.out.println("covariance of Rows: "+covariance.numRows());

		sc.close();
	}
}