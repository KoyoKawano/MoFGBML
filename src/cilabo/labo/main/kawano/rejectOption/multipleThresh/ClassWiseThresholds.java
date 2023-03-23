package cilabo.labo.main.kawano.rejectOption.multipleThresh;


import cilabo.data.DataSet;

public class ClassWiseThresholds implements RejectionBase{

	int thresholdSize;

	public ClassWiseThresholds(DataSet dataset) {

		this.thresholdSize = dataset.getCnum();

	}

	public boolean isReject(ClassificationDataInfo DataInfo, double[] threshold) {

		return DataInfo.getConfidenceValue() < threshold[DataInfo.getWinnerRuleClass()];
	}

	public int getThresholdSize() {

		return this.thresholdSize;
	}
}
