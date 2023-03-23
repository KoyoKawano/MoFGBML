package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import cilabo.data.DataSet;
import cilabo.data.Pattern;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.fuzzy.classifier.operator.classification.factory.SingleWinnerRuleSelection;
import cilabo.fuzzy.rule.RejectedRule;
import cilabo.fuzzy.rule.Rule;
import cilabo.labo.main.kawano.rejectOption.Confidence.Confidence;
import cilabo.labo.main.kawano.rejectOption.Confidence.FirstAndSecondClass;

/**
 * 識別器が識別した情報を格納する．
 * RejectOptionの閾値計算は，パターンの確信度と結論部クラス(CWT)，識別したルールのインデックス(RWT)があれば，できる．
 * 計算時間の関係上，１度計算すればいいので，ここに格納する．
 */
public class ClassificationDataInfo extends DataSet{

	Double confidenceValue;
	Rule WinnerRule;
	int WinnerRuleClass;
	int RuleID;
	boolean isRight;
	Pattern pattern;
	Confidence conf = new FirstAndSecondClass();
	SingleWinnerRuleSelection classification = new SingleWinnerRuleSelection();

	public ClassificationDataInfo(Pattern pattern, RuleBasedClassifier Classifier) {

		this.pattern = pattern;

		this.WinnerRule = classification.classify(Classifier, pattern.getInputVector());

		if(!(WinnerRule instanceof RejectedRule)) {

			this.WinnerRuleClass = WinnerRule
									.getConsequent()
									.getClassLabel().getClassLabel();

			this.RuleID = classification.getWinnerRuleID();

			this.confidenceValue = conf.confidence(Classifier, pattern.getInputVector());

			this.isRight = WinnerRule
							.getConsequent()
							.getClassLabel().toString()
							.equals(pattern.getTrueClass().toString());
		}
		else {
			this.confidenceValue = 0.0;
		}
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