package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import cilabo.data.Pattern;
import cilabo.fuzzy.classifier.RuleBasedClassifier;

public class ClassWiseThresholds implements RejectionBase{

	List<ClassificationDataInfo> classificationInfo;

	int thresholdSize;

	public ClassWiseThresholds(RuleBasedClassifier Classifier, List<Pattern> dataset) {

		this.thresholdSize = dataset.stream()
								.map(x -> x.getTrueClass().getClassLabel())
								.max(Comparator.naturalOrder())
								.orElse(-1) + 1;

		classificationInfo = dataset.stream()
				.map(x-> new ClassificationDataInfo(x, Classifier))
				.collect(Collectors.toList());

	}

	public boolean isReject(ClassificationDataInfo DataInfo, double[] threshold) {

		return DataInfo.getConfidenceValue() < threshold[DataInfo.getWinnerRuleClass()];
	}



	public List<ClassificationDataInfo> getClassificationInfo() {

		return this.classificationInfo;
	}

	public int getThresholdSize() {
		return this.thresholdSize;
	}
}
