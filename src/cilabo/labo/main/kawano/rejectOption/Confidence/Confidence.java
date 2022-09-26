package cilabo.labo.main.kawano.rejectOption.Confidence;

import cilabo.data.InputVector;
import cilabo.fuzzy.classifier.Classifier;

public interface Confidence {

	double confidence(Classifier classifier, InputVector vector);
}
