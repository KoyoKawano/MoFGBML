package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import cilabo.fuzzy.rule.RejectedRule;

public class RejectOptionMetric {

	RejectionBase rejectionBase;

	public RejectOptionMetric(RejectionBase rejectionBase) {

		this.rejectionBase = rejectionBase;
	}


	public double[] initialization() {

		double[] threshold = new double[rejectionBase.getThresholdSize()];
		Arrays.fill(threshold, 0.0);

		return threshold;
	}

	/*
	 * culcAcc(threshold = {0,0,..0}) Reject rate = 0のとき，
	 * Reject Optionなしの精度となるが，識別不能の場合（適合度とルール重みの積が全てのルールに対して0，異なるクラスで等しいとき）
	 * のパターンを除いて計算しているので（誤識別パターンと考えていない）精度が少し高くなる．
	 */
	public double culcAcc(double[] threshold) {

		List<ClassificationDataInfo> acceptData = rejectionBase.getClassificationInfo().stream()
													.filter(x -> !(x.getWinnerRule() instanceof RejectedRule))
													.filter(x -> !rejectionBase.isReject(x, threshold))
													.collect(Collectors.toList());


		if(DoubleStream.of(threshold).allMatch(x -> x == 0.0))
			return (double)acceptData.stream().filter(ClassificationDataInfo::getisRight).count() / rejectionBase.getClassificationInfo().size();

		return (double)acceptData.stream().filter(ClassificationDataInfo::getisRight).count() / acceptData.size();
	}

	public double culcRejectRate(double[] threshold) {

		List<ClassificationDataInfo> classificationInfo = rejectionBase.getClassificationInfo();

		if(DoubleStream.of(threshold).allMatch(x -> x == 0.0))
			return 0.0;
		
		return (double)classificationInfo.stream()
				.filter(x -> !(x.getWinnerRule() instanceof RejectedRule))
				.filter(x -> rejectionBase.isReject(x, threshold))
				.count() / classificationInfo.size();
	}

}
