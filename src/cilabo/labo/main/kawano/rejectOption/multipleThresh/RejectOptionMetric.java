package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import cilabo.fuzzy.rule.RejectedRule;

/*
 * Threshold based Reject Option method:
 * 配列の初期化，Accuracy Reject rateの計算
 *
 */

public class RejectOptionMetric {

	// 閾値の種類 (Single, CWT, RWT)
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

	public double culcAcc(List<ClassificationDataInfo> dataInfo, double[] threshold) {

		List<ClassificationDataInfo> acceptData = dataInfo.stream()
													.filter(x -> !(x.getWinnerRule() instanceof RejectedRule))
													.filter(x -> !rejectionBase.isReject(x, threshold))
													.collect(Collectors.toList());

		if(DoubleStream.of(threshold).allMatch(x -> x == 0.0))
			return (double)acceptData.stream().filter(ClassificationDataInfo::getisRight).count() / rejectionBase.getClassificationInfo().size();

		return (double)acceptData.stream().filter(ClassificationDataInfo::getisRight).count() / acceptData.size();
	}


	public double culcRejectRate(List<ClassificationDataInfo> dataInfo, double[] threshold) {

		if(DoubleStream.of(threshold).allMatch(x -> x == 0.0))
			return 0.0;

		return (double)dataInfo.stream()
				.filter(x -> !(x.getWinnerRule() instanceof RejectedRule))
				.filter(x -> rejectionBase.isReject(x, threshold))
				.count() / dataInfo.size();
	}
}
