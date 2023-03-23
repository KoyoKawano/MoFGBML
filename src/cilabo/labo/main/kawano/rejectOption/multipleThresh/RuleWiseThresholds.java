package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import cilabo.fuzzy.classifier.RuleBasedClassifier;

public class RuleWiseThresholds implements RejectionBase{


	int thresholdSize;


	public RuleWiseThresholds(RuleBasedClassifier Classifier) {

		this.thresholdSize = Classifier.getRuleSet().size();

	}

	public boolean isReject(ClassificationDataInfo DataInfo, double[] threshold) {

		return DataInfo.getConfidenceValue() < threshold[DataInfo.getRuleID()];
	}


	public int getThresholdSize() {

		return this.thresholdSize;
	}
}
