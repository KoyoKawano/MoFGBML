package cilabo.labo.main.kawano.rejectOption.TwoStageRejectOption;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.tuple.Pair;

import cilabo.data.ClassLabel;
import cilabo.data.DataSet;
import cilabo.data.Pattern;

public class KNN implements SecondClassifier{

	private int k;
	private DataSet Dtra;

	public KNN(DataSet Dtra, int k) {
		setK(k);
		setDataset(Dtra);
	}

	public void fit(DataSet Dtra, int k) {
		setK(k);
		setDataset(Dtra);
	}


	public ClassLabel predict(Pattern pattern) {

		List<Pair<Pattern, Double>> rank = Dtra.getPatterns().stream()
											.map(x -> Pair.of(x, EuclideanDistance(x, pattern)))
											.sorted(Comparator.comparing(p -> p.getRight()))
											.collect(Collectors.toList());


		int[] label = new int[Dtra.getCnum()];

		for(int i = 0; i < k; i++)

			label[rank.get(i).getLeft().getTrueClass().getClassLabel()] ++;

		int max = IntStream.of(label).max().getAsInt();

		List<Integer> argmax = new ArrayList<Integer>();

		for(int i = 0; i < label.length; i++) {

			if(label[i] == max)
				argmax.add(i);

		}

		if(argmax.size() != 1)
			return null;

		ClassLabel result = new ClassLabel();

		result.addClassLabel(argmax.get(0));

		return result;
	}

	public void setK(int k) {
		this.k = k;
	}

	public void setDataset(DataSet Dtra) {
		this.Dtra = Dtra;
	}

	public double EuclideanDistance(Pattern x, Pattern y) {

		double sum = 0.0;

		for(int i = 0; i < x.getInputVector().getVector().length; i++)

			sum += Math.pow(x.getDimValue(i) - y.getDimValue(i), 2);

		return Math.sqrt(sum);
	}
}
