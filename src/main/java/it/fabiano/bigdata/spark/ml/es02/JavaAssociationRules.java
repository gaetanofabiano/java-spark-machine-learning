
package it.fabiano.bigdata.spark.ml.es02;
/**
* Java-Spark-Training-Course
*
* @author  Gaetano Fabiano
* @version 1.0.0
* @since   2019-07-19 
*/

import java.util.Arrays;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset;
// $example off$

import org.apache.spark.SparkConf;

public class JavaAssociationRules {

  public static void main(String[] args) {

    SparkConf sparkConf = new SparkConf().setAppName("JavaAssociationRulesExample").setMaster("local[*]");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    
    JavaRDD<FPGrowth.FreqItemset<String>> freqItemsets = sc.parallelize(Arrays.asList(
      new FreqItemset<>(new String[] {"a"}, 15L),
      new FreqItemset<>(new String[] {"b"}, 35L),
      new FreqItemset<>(new String[] {"a", "b"}, 12L)
    ));

    AssociationRules arules = new AssociationRules()
      .setMinConfidence(0.1);
    JavaRDD<AssociationRules.Rule<String>> results = arules.run(freqItemsets);

    for (AssociationRules.Rule<String> rule : results.collect()) {
      System.out.println(
        rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
    }
   
    
    sc.stop();
    sc.close();
  }
}
