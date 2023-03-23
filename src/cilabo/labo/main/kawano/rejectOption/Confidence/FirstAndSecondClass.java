package cilabo.labo.main.kawano.rejectOption.Confidence;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import cilabo.data.InputVector;
import cilabo.fuzzy.classifier.Classifier;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.fuzzy.rule.Rule;

/**
 *
 * @author kawano
 *
 * 2つのルールの競合度合いによる確信度
 */
public class FirstAndSecondClass implements Confidence{


		public FirstAndSecondClass() {
			// TODO 自動生成されたコンストラクター・スタブ
		}

		/**
		 * 確信度を返す関数
		 * 識別不能な場合，-1.0を返す（必ずrejectされる）
		 */
		public double confidence(Classifier classifier, InputVector vector) {

			if(classifier.getClass() != RuleBasedClassifier.class) return -1.0;

			List<Rule> ruleSet = ((RuleBasedClassifier)classifier).getRuleSet();

			List<Pair<Rule, Double>> ruleValue = new ArrayList<Pair<Rule, Double>>();

			for(Rule rule : ruleSet) {

				double membership = rule.getAntecedent().getCompatibleGrade(vector.getVector());

				double CF = rule.getConsequent().getRuleWeight().getRuleWeight();

				double value = membership * CF;

				ruleValue.add(Pair.of(rule, value));
			}

			Pair<Rule, Double> winnerRule = ruleValue.stream()
					.max(Comparator.comparingDouble(o -> o.getRight()))
					.orElse(Pair.of(null));

			Pair<Rule, Double> secondWinnerRule = ruleValue.stream()
					.filter(x -> !x.getLeft().getConsequent().getClassLabel().toString().equals(winnerRule.getLeft().getConsequent().getClassLabel().toString()))
					.max(Comparator.comparingDouble(o -> o.getRight()))
					.orElse(null);


			return (winnerRule.getRight() - secondWinnerRule.getRight()) / winnerRule.getRight();
		}

}
