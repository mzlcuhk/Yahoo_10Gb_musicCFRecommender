package recommender;

import java.io.Serializable;
import java.util.Comparator;

import scala.Tuple2;

public class TupleComparator implements Comparator<Tuple2<Integer, Double>>, Serializable {
	public int compare(Tuple2<Integer, Double> o1,
			Tuple2<Integer, Double> o2) {
		// TODO Auto-generated method stub
		double difference = o1._2() - o2._2();
		if (difference < 0) {
			return 1;
		} else if (difference == 0) {
			return 0;
		} else {
			return -1;
		}
	}

	
}
