package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.Arrays;

import cilabo.fuzzy.classifier.operator.classification.factory.SingleWinnerRuleSelection;

public class EstimateThreshold {

	/********** fields ************************/

	SingleWinnerRuleSelection classification = new SingleWinnerRuleSelection();

	RejectionBase rejectionBase;
	int kmax;
	double deltaT;
	double Rmax;
	double[] threshold;

	/*******************************************/


	/********** constructor ************************/

	public EstimateThreshold(RejectionBase rejectionBase, int kmax, double deltaT, double Rmax){

		this.rejectionBase = rejectionBase;
		this.kmax = kmax;
		this.deltaT = deltaT;
		this.Rmax = Rmax;
	}
	/*******************************************/



	public double[] run() {

		/*** Initiallization ***/

		RejectOptionMetric metric = new RejectOptionMetric(rejectionBase);

		threshold = metric.initialization();

		double[] bestThresh = metric.initialization();

		double bestAcc = metric.culcAcc(threshold);

		double bestReject = 0.0;

		boolean isLoop = true;

		/*************************/

		while(isLoop) {

			double prebest = bestAcc;
			double prereject = bestReject;
			double bestValue = 0.0;

			for(int id = 0; id < threshold.length; id++) {

				for(int k = 0; k <= kmax; k++) {

					double[] thresh = threshold.clone();

					thresh[id] += k * deltaT;

					double acc = metric.culcAcc(thresh);

					double rej = metric.culcRejectRate(thresh);
//					System.out.println("acc = " + String.valueOf(acc) + " RR = " + String.valueOf(rej));

					//Rmaxを超えない棄却割合でaccを改善した場合，
					//小さい棄却の変化で大きいaccの増加を達成する閾値に更新
					if(acc > prebest && rej < this.Rmax) {

						double value = (acc - prebest) / (rej - prereject);

						if(value > bestValue) {
							bestAcc = acc;
							bestReject = rej;
							bestThresh = thresh.clone();
							bestValue = value;
						}
					}
				}
			}
			//探索範囲でaccを改善できなかった場合，終了
			if(Arrays.equals(threshold, bestThresh))
				isLoop = false;

			//探索範囲内で改善できた場合はもう一度探索
			else
				threshold = bestThresh.clone();

		}

		double[] result = new double[] {bestAcc, bestReject};

		return result;
	}

	public double[] getThreshold() {
		return threshold;
	}
}
