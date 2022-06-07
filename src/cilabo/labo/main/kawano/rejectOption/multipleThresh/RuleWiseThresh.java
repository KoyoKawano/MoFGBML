package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.tuple.Triple;

import cilabo.data.Pattern;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.fuzzy.classifier.operator.classification.factory.SingleWinnerRuleSelection;

public class RuleWiseThresh implements MultipleThreshold {

	int kmax;
	double deltaT;
	double Rmax;
	double[] threshold;

	public RuleWiseThresh(int kmax, double deltaT, double Rmax) {

		this.kmax = kmax;
		this.deltaT = deltaT;
		this.Rmax = Rmax;
	}

	public double[] estimateThresh(RuleBasedClassifier Classifier, List<Pattern> dataset) {


		SingleWinnerRuleSelection classification = new SingleWinnerRuleSelection();

		//各入力に対する確信度と勝者ルール，正誤のペアのTripleのリスト
		List<Triple<Double, Integer, Boolean>> confidence = new ArrayList<Triple<Double, Integer,Boolean>>();

		for(Pattern pattern : dataset) {

			classification.classify(Classifier, pattern.getInputVector());

			double con = classification.getConfidence();

			int ruleID = classification.getWinnerRuleID();

			boolean tf = classification
							.classify(Classifier, pattern.getInputVector())
							.getConsequent()
							.getClassLabel().getClassLabel() == pattern.getTrueClass().getClassLabel();

			confidence.add(Triple.of(con, ruleID, tf));
		}


		int ruleNum = Classifier.getRuleLength() + 1;

		//初期化
		threshold = new double[ruleNum];
		Arrays.fill(threshold, 0.0);
		boolean isLoop = true;
		double[] bestthresh = new double[ruleNum];
		double bestacc = 0.0;
		double reject = 1.0;

		while(isLoop) {

			for(int id = 0; id < ruleNum; id++) {

				for(int k = 0; k <= kmax; k++) {

					double[] thresh = Arrays.copyOf(threshold, ruleNum);

					thresh[id] += k * deltaT;

					//確信度が閾値以上の入力の内，正答した割合
					double acc = (double)confidence.stream()
							.filter(x -> x.getLeft() >= thresh[x.getMiddle()] && x.getRight())
							.count() /
							confidence.stream()
							.filter(x -> x.getLeft() >= thresh[x.getMiddle()])
							.count();

					//入力全体の内，確信度が閾値より低く棄却された割合
					double rej = (double)confidence.stream()
							.filter(x -> x.getLeft() < thresh[x.getMiddle()])
							.count() /
							confidence.size();

					//Rmaxを超えない棄却割合でaccを改善した場合 または，
					//小さい棄却で同じaccを達成するrejだった場合，更新を行う
					if((acc > bestacc && rej < Rmax) || (acc == bestacc && rej < reject)) {
						bestacc = acc;
						reject = rej;
						bestthresh = Arrays.copyOf(thresh, ruleNum);
					}
				}
			}

				//探索範囲でaccを改善できなかった場合，終了
				if(Arrays.equals(threshold, bestthresh))
					isLoop = false;

				//探索範囲内で改善できた場合はもう一度探索
				else
					threshold = Arrays.copyOf(bestthresh, ruleNum);

		}

		double[] result = new double[] {bestacc, reject};

		return result;
	}
}
