package cilabo.labo.main.kawano.rejectOption.multipleThresh;

/**
 *
 * @author kawano
 *
 * 閾値ベースの棄却オプションのインターフェース
 */
public interface RejectionBase {

	/**
	 * １つのパターンを棄却するかを判定する関数
	 * @param dataInfo : パターンの識別情報(識別されたクラス，ルールIDなど)
	 * @param threshold: 閾値 List<Double>
	 * @return boolean (True : 棄却，False : 棄却しない)
	 */
	boolean isReject(ClassificationDataInfo dataInfo, double[] threshold);


	/**
	 *棄却オプションの閾値の数を取得する関数
	 *
	 * @return 閾値の長さ int
	 */
	int getThresholdSize();

}
