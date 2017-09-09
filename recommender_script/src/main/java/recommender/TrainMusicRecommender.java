package recommender;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.rdd.RDD;
public class TrainMusicRecommender {
	@SuppressWarnings("unchecked")
	public static void main(String[] args) {
		if (args.length!=2) {
			System.err
					.println("Usage:\n\t TrainMusicRecommender <training files path> <testing files path>");
			System.exit(2);
		}
		String trainFiles = args[0].trim();
		String testFiles = args[1].trim();
		SparkConf conf = new SparkConf().setAppName("Music Recommender");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		JavaRDD<String> trainData = jsc.textFile(trainFiles);
		JavaRDD<Rating> ratings = trainData.map(new Function<String,Rating>(){

			public Rating call(String str) throws Exception {
				String[] splitStr = str.split("\t");
				return new Rating(Integer.parseInt(splitStr[0]),Integer.parseInt(splitStr[1]),Integer.parseInt(splitStr[2]));
			}
			
		});
		List<String> outputStrs = new ArrayList<String>();
		int rank = 60;
		JavaRDD<Tuple2<Double, Double>> ratesAndPreds=null;
		MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, 10,0.1);
		JavaRDD<String> testData = jsc.textFile(testFiles);
		JavaRDD<Rating> testRatings = testData
				.map(new Function<String, Rating>() {
					public Rating call(String str) throws Exception {
						String[] splitStr = str.split("\t");
						return new Rating(Integer
								.parseInt(splitStr[0]), Integer
								.parseInt(splitStr[1]), Integer
								.parseInt(splitStr[2]));
					}
				});
		// Evaluate the model on rating data
		JavaRDD<Tuple2<Object, Object>> userSongs = testRatings
				.map(new Function<Rating, Tuple2<Object, Object>>() {
					public Tuple2<Object, Object> call(Rating r)
							throws Exception {
						return new Tuple2<Object, Object>(r.user(),
								r.product());
					}
				});
		JavaRDD<Rating> predictionsJavaRDD = model.predict(
				JavaRDD.toRDD(userSongs)).toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Integer, Integer>, Double>> predictionsTupleJavaRDD = predictionsJavaRDD
				.map(new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {

					public Tuple2<Tuple2<Integer, Integer>, Double> call(
							Rating r) throws Exception {
						return new Tuple2<Tuple2<Integer, Integer>, Double>(
								new Tuple2<Integer, Integer>(r
										.user(), r.product()), r
										.rating());
					}
				});
		JavaPairRDD predictionsJavaPairRDD = JavaPairRDD
				.fromJavaRDD(predictionsTupleJavaRDD);
		ratesAndPreds = 
				JavaPairRDD
				.fromJavaRDD(
						testRatings
								.map(new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
									public Tuple2<Tuple2<Integer, Integer>, Double> call(
											Rating r)
											throws Exception {
										return new Tuple2<Tuple2<Integer, Integer>, Double>(
												new Tuple2<Integer, Integer>(
														r.user(),
														r.product()),
												r.rating());
									}
								})).join(predictionsJavaPairRDD).values();
		double testMSE = ratesAndPreds.mapToDouble(
				new DoubleFunction<Tuple2<Double, Double>>() {

					public double call(Tuple2<Double, Double> t)
							throws Exception {
						double err = t._1() - t._2();
						return err * err;
					}
				}).mean();
		model.save(jsc.sc(), "s3://ds504/users/yangzheng/recommenderModel");
		List<Tuple2<Double,Double>> ratesAndPredsList = ratesAndPreds.take(50);
		System.out.println("For rank = "+rank+" Test Mean Squared Error = " + testMSE);
		outputStrs.add("When rank = "+rank+" Test Mean Squared Error = " + testMSE);
		JavaRDD<String> outputRDD =jsc.parallelize(outputStrs);
		outputRDD.saveAsTextFile("s3://ds504/users/yangzheng/output");
		
		
	}
}
