package cilabo.gbml.operator.heuristic;

import java.util.ArrayList;

import cilabo.data.DataSet;
import cilabo.data.Pattern;
import cilabo.fuzzy.knowledge.Knowledge;

public class PatternBaseRuleGenerationBuilder {

	int H;
	Knowledge problem;
	ArrayList<Pattern> train;

	public PatternBaseRuleGenerationBuilder(int H, Knowledge problem, ArrayList<Pattern> train) {

		this.H = H;
		this.problem = problem;
		this.train = train;
	}


	public PatternBaseRuleGeneration build() {

		if(H == 1)
			return new HeuristicRuleGeneration(problem);

		DataSet Train = new DataSet();
		Train.setPattern(train);
		return new MultipatternHeuristicRuleGeneration(problem, H, Train);
	}
}
