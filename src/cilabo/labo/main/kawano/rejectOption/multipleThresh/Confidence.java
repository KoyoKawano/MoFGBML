package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.List;

import cilabo.data.InputVector;
import cilabo.fuzzy.classifier.Classifier;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.fuzzy.rule.Rule;

public class Confidence {

	//確信度を返す関数
	//識別不能な場合，-1.0を返す（必ずrejectされる）

		static public double confidence(Classifier classifier, InputVector vector) {
			if(classifier.getClass() != RuleBasedClassifier.class) return -1.0;

			List<Rule> ruleSet = ((RuleBasedClassifier)classifier).getRuleSet();

			boolean canClassify = true;
			double max = -Double.MAX_VALUE;
			int winner = 0;
			for(int q = 0; q < ruleSet.size(); q++) {
				Rule rule = ruleSet.get(q);
				double membership = rule.getAntecedent().getCompatibleGrade(vector.getVector());
				double CF = rule.getConsequent().getRuleWeight().getRuleWeight();

				double value = membership * CF;
				if(value > max) {
					max = value;
					winner = q;
					canClassify = true;
				}
				else if(value == max) {
					Rule winnerRule = ruleSet.get(winner);
					// "membership*CF"が同値 かつ 結論部クラスが異なる
					if(!rule.getConsequent().getClassLabel().toString().equals(winnerRule.getConsequent().getClassLabel().toString())) {
						canClassify = false;
					}
				}
			}

			if(canClassify && max > 0) {
				return max;
			}
			else {
				return -1.0;
			}
		}
}
