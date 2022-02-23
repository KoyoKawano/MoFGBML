package cilabo.gbml.operator.heuristic.select;

import java.util.List;

import cilabo.data.Pattern;

public interface SelectSupportPattern {

	// @param int H : the number of patterns to use heuristic rule generation
	//                one pattern is base pattern, the other(H-1) patterns are support pattern.
	//
	// @param Pattern basePattern

	List<Pattern> execute(int H, Pattern basePattern);
}