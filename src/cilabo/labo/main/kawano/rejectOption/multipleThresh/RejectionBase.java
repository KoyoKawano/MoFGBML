package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.List;


public interface RejectionBase {

	boolean isReject(ClassificationDataInfo dataInfo, double[] threshold);

	List<ClassificationDataInfo> getClassificationInfo();

	int getThresholdSize();

}
