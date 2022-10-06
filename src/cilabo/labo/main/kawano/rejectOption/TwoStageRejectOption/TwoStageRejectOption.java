package cilabo.labo.main.kawano.rejectOption.TwoStageRejectOption;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import cilabo.data.Pattern;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.fuzzy.rule.RejectedRule;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.ClassificationDataInfo;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.EstimateThreshold;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.RejectionBase;

public class TwoStageRejectOption {

	RejectionBase rejectionBase;
	SecondClassifier secondClassifier;
	List<ClassificationDataInfo> classificationInfo;

	double[] threshold;

	public TwoStageRejectOption(RejectionBase rejectionBase, SecondClassifier secondClassifier, RuleBasedClassifier Classifier, List<Pattern> dataset) {

		this.rejectionBase = rejectionBase;
		this.secondClassifier = secondClassifier;

		classificationInfo = dataset.stream()
				.map(x-> new ClassificationDataInfo(x, Classifier))
				.collect(Collectors.toList());

	}

	public boolean isReject(ClassificationDataInfo dataInfo) {

		boolean SecondStageJudge = dataInfo.getWinnerRuleClass() != secondClassifier.predict(dataInfo.getPattern()).getClassLabel();

		return rejectionBase.isReject(dataInfo, threshold) && SecondStageJudge;
	}

	public double culcAcc() {

		List<ClassificationDataInfo> acceptData = classificationInfo.stream()
													.filter(x -> !(x.getWinnerRule() instanceof RejectedRule))
													.filter(x -> !isReject(x))
													.collect(Collectors.toList());

		if(DoubleStream.of(threshold).allMatch(x -> x == 0.0))
			return (double)acceptData.stream().filter(ClassificationDataInfo::getisRight).count() / rejectionBase.getClassificationInfo().size();

		return (double)acceptData.stream().filter(ClassificationDataInfo::getisRight).count() / acceptData.size();
	}

	public double culcRejectOption() {

		List<ClassificationDataInfo> classificationInfo = rejectionBase.getClassificationInfo();

		if(DoubleStream.of(threshold).allMatch(x -> x == 0.0))
			return 0.0;

		return (double)classificationInfo.stream()
				.filter(x -> !(x.getWinnerRule() instanceof RejectedRule))
				.filter(x -> rejectionBase.isReject(x, threshold))
				.count() / classificationInfo.size();
	}

	public void setThreshold(double[] threshold) {

		this.threshold = threshold;
	}

	public void setThreshold(EstimateThreshold estimateThreshold, RuleBasedClassifier Classifier, List<Pattern> dataset) {

		estimateThreshold.run(Classifier, dataset);
		this.threshold = estimateThreshold.getThreshold();
	}
}