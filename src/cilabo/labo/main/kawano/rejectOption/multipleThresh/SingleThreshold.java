package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.List;
import java.util.stream.Collectors;

import cilabo.data.DataSet;
import cilabo.fuzzy.classifier.RuleBasedClassifier;

public class SingleThreshold implements RejectionBase{

	List<ClassificationDataInfo> classificationInfo;

	int thresholdSize = 1;

	public SingleThreshold(RuleBasedClassifier Classifier, DataSet dataset) {

		classificationInfo = dataset.getPatterns().stream()
				.map(x-> new ClassificationDataInfo(x, Classifier))
				.collect(Collectors.toList());

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
