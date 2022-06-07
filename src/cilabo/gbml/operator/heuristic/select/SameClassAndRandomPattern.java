package cilabo.gbml.operator.heuristic.select;

import java.util.ArrayList;
import java.util.List;

import cilabo.data.ClassLabel;
import cilabo.data.Pattern;
import cilabo.utility.GeneralFunctions;
import cilabo.utility.Random;

public class SameClassAndRandomPattern implements SelectSupportPattern {

	private ArrayList<Pattern> dataset;

	public SameClassAndRandomPattern(ArrayList<Pattern> dataset) {

		this.dataset = dataset;

	}

	public List<Pattern> execute(int H, Pattern basePattern){

		ClassLabel baseClass = basePattern.getTrueClass();

		int candidateNum = (int) dataset.stream()
							.filter(x-> x.getTrueClass().getClassLabel() == baseClass.getClassLabel())
							.count();

		Integer[] randIndexlist = GeneralFunctions.samplingWithout(candidateNum, candidateNum, Random.getInstance().getGEN());

		int baseID = basePattern.getID();

		List<Pattern> supportPattern = new ArrayList<>();

		for(int i = 0; i < (H - 1); i++) {

			if(dataset.get(randIndexlist[i]).getID() != baseID) {
				supportPattern.add(dataset.get(randIndexlist[i]));
			}
			else {
				supportPattern.add(dataset.get(randIndexlist[H - 1]));
			}
		}

		return supportPattern;
	}
}
