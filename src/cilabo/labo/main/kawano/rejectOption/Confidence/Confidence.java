package cilabo.labo.main.kawano.rejectOption.Confidence;

import cilabo.data.InputVector;
import cilabo.fuzzy.classifier.Classifier;

/**
 *
 * @author kawano
 *
 * ファジィ識別器の確信度のインターフェース
 */
public interface Confidence {

	double confidence(Classifier classifier, InputVector vector);
}
