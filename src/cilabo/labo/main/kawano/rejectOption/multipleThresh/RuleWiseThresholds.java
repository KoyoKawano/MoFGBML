package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.List;
import java.util.stream.Collectors;

import cilabo.data.Pattern;
import cilabo.fuzzy.classifier.RuleBasedClassifier;

public class RuleWiseThresholds implements RejectionBase{


	List<ClassificationDataInfo> classificationInfo;

	int thresholdSize;


	public RuleWiseThresholds(RuleBasedClassifier Classifier, List<Pattern> dataset) {

		this.thresholdSize = Classifier.getRuleLength();

		this.classificationInfo = dataset.stream()
				.map(x-> new ClassificationDataInfo(x, Classifier))
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
