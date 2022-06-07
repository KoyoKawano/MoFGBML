package cilabo.labo.main.kawano.rejectOption.multipleThresh;

import java.util.List;

import cilabo.data.Pattern;
import cilabo.fuzzy.classifier.RuleBasedClassifier;

public interface MultipleThreshold {

	double[] estimateThresh(RuleBasedClassifier Classifier, List<Pattern> dataset);

}
