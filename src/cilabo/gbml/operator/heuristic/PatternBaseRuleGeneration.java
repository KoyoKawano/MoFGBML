package cilabo.gbml.operator.heuristic;

import java.util.List;

import cilabo.data.Pattern;
import cilabo.fuzzy.knowledge.Knowledge;
import cilabo.fuzzy.rule.antecedent.Antecedent;

public interface PatternBaseRuleGeneration{

	List<Antecedent> execute(List<Pattern> erroredPatterns);
	
	Antecedent ruleGenerate(Pattern Pattern);
	
	Knowledge getKnowledge();
}
