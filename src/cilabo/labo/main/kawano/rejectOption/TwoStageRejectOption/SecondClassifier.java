package cilabo.labo.main.kawano.rejectOption.TwoStageRejectOption;

import cilabo.data.ClassLabel;
import cilabo.data.Pattern;

public interface SecondClassifier {

	ClassLabel predict(Pattern x);
}
