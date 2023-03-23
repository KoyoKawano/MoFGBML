package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.List;

public class SingleThreshold implements RejectionBase{

	List<ClassificationDataInfo> classificationInfo;

	int thresholdSize = 1;

	public SingleThreshold() {


	}

	public boolean isReject(ClassificationDataInfo DataInfo, double[] threshold) {

		return DataInfo.getConfidenceValue() < threshold[0];
	}

	public List<ClassificationDataInfo> getClassificationInfo() {

		return this.classificationInfo;
	}

	public int getThresholdSize() {
		return this.thresholdSize;
	}

}
