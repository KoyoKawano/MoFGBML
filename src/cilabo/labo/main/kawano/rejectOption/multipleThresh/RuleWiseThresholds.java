package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.List;
import java.util.stream.Collectors;

import cilabo.data.DataSet;
import cilabo.fuzzy.classifier.RuleBasedClassifier;

public class RuleWiseThresholds implements RejectionBase{


	List<ClassificationDataInfo> classificationInfo;

	int thresholdSize;


	public RuleWiseThresholds(RuleBasedClassifier Classifier, DataSet dataset) {

		this.thresholdSize = Classifier.getRuleSet().size();

		this.classificationInfo = dataset.getPatterns().stream()
				.map(x -> new ClassificationDataInfo(x, Classifier))
				.collect(Collectors.toList());
	}

	public boolean isReject(ClassificationDataInfo DataInfo, double[] threshold) {

		return DataInfo.getConfidenceValue() < threshold[DataInfo.getRuleID()];
	}


	public List<ClassificationDataInfo> getClassificationInfo() {

		return this.classificationInfo;
	}

	public int getThresholdSize() {

		return this.thresholdSize;
	}
}
