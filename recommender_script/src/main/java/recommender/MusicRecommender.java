package recommender;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.RDD;

import scala.Tuple2;
import scala.reflect.reify.phases.Calculate;

public class MusicRecommender  implements Serializable{
	public static void main(String[] args) {
		int rank = 60;
		SparkConf conf = new SparkConf().setAppName("Music Recommender").setMaster("local[*]");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		jsc.setLogLevel("ERROR");
		MatrixFactorizationModel model = MatrixFactorizationModel.load(jsc.sc(),"recommenderModel");
		MusicRecommender s = new MusicRecommender();
		s.recommendSimilarSongs(model, 113);
		s.recommendSongsToUser(model, 200115, 10);
	}
	public void recommendSimilarSongs(MatrixFactorizationModel model,int songID) {
		RDD<Tuple2<Object, double[]>> productFeatures= model.productFeatures();
		JavaRDD<Tuple2<Object, double[]>> javaRDDProductFeatures = productFeatures.toJavaRDD();
		JavaPairRDD<Integer, double[]> songFeaturesPairRDD = javaRDDProductFeatures.mapToPair(new PairFunction<Tuple2<Object, double[]>,Integer,double[] >(){

			public Tuple2<Integer, double[]> call(Tuple2<Object, double[]> t)
					throws Exception {
				
				return new Tuple2<Integer, double[]>((Integer)t._1(),t._2());
			}
			
		});
		final double[] songFeatures = songFeaturesPairRDD.lookup(songID).get(0);
		JavaPairRDD<Integer, Double> similarSongs = songFeaturesPairRDD.mapToPair(new PairFunction<Tuple2<Integer,double[]>,Integer,Double>(){
			
			public Tuple2<Integer, Double> call(Tuple2<Integer, double[]> t)
					throws Exception {
				double cosSimilarity = calculateCosSimilarity(songFeatures, t._2());
				return new Tuple2<Integer, Double>(t._1(),cosSimilarity);
			}
		});
		Comparator<Tuple2<Integer, Double>> comp = new TupleComparator();
		List<Tuple2<Integer, Double>> top20SimilarySongs = similarSongs.takeOrdered(20, comp);
		System.out.println("\n===Top 20 similar songs for Song "+songID+" with corresponding ID and cosine similarity===");
		for (Tuple2<Integer, Double> t : top20SimilarySongs) {
			System.out.println(String.format("Song ID: %d\tCosine Similarity: %f", t._1(),t._2()));
		}
	}
	
	public double calculateCosSimilarity(double[] songFeatures1, double[] songFeatures2) {
		
		if(songFeatures1.length != songFeatures2.length) {
			throw new RuntimeException("Two double vectors with different lengths!");
		}
		double innerProduct = 0;
		double sumOfSquares1 = 0;
		double sumOfSquares2 = 0;
		for(int i = 0; i <songFeatures1.length; i++) {
			double value1 = songFeatures1[i];
			double value2 = songFeatures2[i];
			innerProduct += value1 * value2;
			sumOfSquares1 += value1 * value1;
			sumOfSquares2 += value2 * value2;
			
		}
		return innerProduct / Math.sqrt(sumOfSquares1 * sumOfSquares2);
	}
	
	public void recommendSongsToUser(MatrixFactorizationModel model, int userID,int amount) {
		Rating[] ratings = model.recommendProducts(userID, amount);
		System.out.println("\n===Top "+amount+" songs recommended to user "+userID+"===");
		for (Rating r : ratings) {
			System.out.println(String.format("Song ID: %d",r.product()));
		}
	}
}
