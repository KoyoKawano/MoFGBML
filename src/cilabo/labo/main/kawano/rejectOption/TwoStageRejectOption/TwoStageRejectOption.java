package cilabo.labo.main.kawano.rejectOption.TwoStageRejectOption;

import java.util.List;

import cilabo.labo.main.kawano.rejectOption.multipleThresh.ClassificationDataInfo;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.RejectionBase;

public class TwoStageRejectOption implements RejectionBase{

	RejectionBase rejectionBase;
	SecondClassifier secondClassifier;
	List<ClassificationDataInfo> classificationInfo;

	double[] threshold;

	public TwoStageRejectOption(RejectionBase rejectionBase, SecondClassifier secondClassifier) {

		this.rejectionBase = rejectionBase;
		this.secondClassifier = secondClassifier;

	}

	public boolean isReject(ClassificationDataInfo dataInfo, double[] threshold) {

		boolean isRejectSecondStage = dataInfo.getWinnerRuleClass() != secondClassifier.predict(dataInfo.getPattern()).getClassLabel();

		return rejectionBase.isReject(dataInfo, threshold) && isRejectSecondStage;
	}

	public int getThresholdSize() {

		return this.rejectionBase.getThresholdSize();
	}
}