package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import cilabo.data.Pattern;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.fuzzy.classifier.operator.classification.factory.SingleWinnerRuleSelection;

public class Single implements MultipleThreshold{

	int kmax;
	double deltaT;
	double Rmax;
	double[] threshold;

	public Single(int kmax, double deltaT, double Rmax) {

		this.kmax = kmax;
		this.deltaT = deltaT;
		this.Rmax = Rmax;
	}

	public double[] estimateThresh(RuleBasedClassifier Classifier, List<Pattern> dataset) {

		threshold = new double[] {0.0};
		SingleWinnerRuleSelection classification = new SingleWinnerRuleSelection();

		//各入力に対する確信度と正誤のペアのリスト
		List<Pair<Double, Boolean>> confidence = new ArrayList<Pair<Double,Boolean>>();

		for(Pattern pattern : dataset) {

			double con = Confidence.confidence(Classifier, pattern.getInputVector());


			boolean tf = classification
							.classify(Classifier, pattern.getInputVector())
							.getConsequent()
							.getClassLabel().getClassLabel() == pattern.getTrueClass().getClassLabel();

			confidence.add(Pair.of(con, tf));
		}

		//初期化
		boolean isLoop = true;
		double bestthresh = 0.0;
		double bestacc = 0.0;
		double reject = 1.0;

		while(isLoop) {

			for(int k = 0; k <= kmax; k++) {

				double thresh = threshold[0] + k * deltaT;


				//確信度が閾値以上の入力の内，正答した割合
				double acc = (double)confidence.stream()
						.filter(x -> x.getLeft() >= thresh && x.getRight())
						.count() /
						(double)confidence.stream()
						.filter(x -> x.getLeft() >= thresh)
						.count();

				//入力全体の内，確信度が閾値より低く棄却された割合
				double rej = (double)confidence.stream()
						.filter(x -> x.getLeft() < thresh)
						.count() /
						confidence.size();

//				System.out.println(acc);
//				System.out.println(reject);
				//Rmaxを超えない棄却割合でaccを改善した場合 または，
				//小さい棄却で同じaccを達成するrejだった場合，更新を行う
				if(acc > bestacc && rej < Rmax) {
					System.out.println("update");
					bestacc = acc;
					reject = rej;
					bestthresh = thresh;
				}
			}

			//探索範囲でaccを改善できなかった場合，終了
			if(bestthresh == threshold[0])
				isLoop = false;

			//探索範囲内で改善できた場合はもう一度探索
			else
				threshold[0] = bestthresh;

		}


		double[] result = new double[] {bestacc, reject};

		return result;
	}

}
