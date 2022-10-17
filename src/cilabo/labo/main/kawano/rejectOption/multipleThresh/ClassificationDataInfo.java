package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import cilabo.data.DataSet;
import cilabo.data.Pattern;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.fuzzy.classifier.operator.classification.factory.SingleWinnerRuleSelection;
import cilabo.fuzzy.rule.RejectedRule;
import cilabo.fuzzy.rule.Rule;
import cilabo.labo.main.kawano.rejectOption.Confidence.Confidence;
import cilabo.labo.main.kawano.rejectOption.Confidence.FirstAndSecondClass;

public class ClassificationDataInfo extends DataSet{

	Double confidenceValue;
	Rule WinnerRule;
	int WinnerRuleClass;
	int RuleID;
	boolean isRight;
	Pattern pattern;
	Confidence delta = new FirstAndSecondClass();

	public ClassificationDataInfo(Pattern pattern, RuleBasedClassifier Classifier) {

		SingleWinnerRuleSelection classification = new SingleWinnerRuleSelection();

		this.pattern = pattern;

		this.WinnerRule = classification.classify(Classifier, pattern.getInputVector());

		if(!(WinnerRule instanceof RejectedRule)) {

			this.WinnerRuleClass = WinnerRule
					.getConsequent()
					.getClassLabel().getClassLabel();

			this.RuleID = classification.getWinnerRuleID();

			this.confidenceValue = delta.confidence(Classifier, pattern.getInputVector());

			this.isRight = WinnerRule
							.getConsequent()
							.getClassLabel().toString()
							.equals(pattern.getTrueClass().toString());
		}
		else {
			this.confidenceValue = 0.0;
		}
	}

	public String toString() {

		String str = this.pattern.toString();
		str += ", conf :" + String.valueOf(confidenceValue);
		str += ", TorF : " + String.valueOf(this.isRight);

		return str;
 	}

	public Double getConfidenceValue() {
		return this.confidenceValue;
	}

	public Rule getWinnerRule() {
		return this.WinnerRule;
	}

	public int getWinnerRuleClass() {
		return this.WinnerRuleClass;
	}

	public int getRuleID() {
		return this.RuleID;
	}

	public boolean getisRight() {
		return this.isRight;
	}

	public Pattern getPattern() {
		return pattern;
	}
}