package cilabo.gbml.operator.heuristic.select;

import java.util.List;

import cilabo.data.Pattern;

public interface SelectSupportPattern {

	List<Pattern> execute(Pattern basePattern);
}