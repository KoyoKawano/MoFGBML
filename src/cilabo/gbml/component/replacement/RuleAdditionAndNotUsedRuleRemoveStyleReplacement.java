package cilabo.gbml.component.replacement;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.uma.jmetal.component.evaluation.Evaluation;
import org.uma.jmetal.component.replacement.Replacement;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.integersolution.IntegerSolution;
import org.uma.jmetal.util.comparator.ObjectiveComparator;

import cilabo.gbml.component.evaluation.MichiganEvaluation;
import cilabo.gbml.solution.MichiganSolution;
import cilabo.gbml.solution.util.attribute.NumberOfWinner;

/**
 * 精度重視型Michigan style FGBMLに使用
 */
//使用する場合，個体群の評価もしているので，環境選択の後個体群評価しないように気を付ける．
//(評価しても，winnerNum = 0を削除しているだけなので,評価が変わらない)
public class RuleAdditionAndNotUsedRuleRemoveStyleReplacement
	implements Replacement<IntegerSolution> {

	private Evaluation<IntegerSolution> evaluation = new MichiganEvaluation<IntegerSolution>();
	private Problem<IntegerSolution> problem;

	public RuleAdditionAndNotUsedRuleRemoveStyleReplacement(Problem<IntegerSolution> problem) {
		this.problem = problem;
	}
	public List<IntegerSolution> replace(List<IntegerSolution> currentList, List<IntegerSolution> offspringList) {

		// 親個体をfitness順にソートする
		Collections.sort(currentList,
						 new ObjectiveComparator<IntegerSolution>(0, ObjectiveComparator.Ordering.DESCENDING));

		// Add rules
		currentList.addAll(offspringList);
//		for(int i = 0; i < offspringList.size(); i++) {
//			currentList.add(offspringList.get(i));
//		}

		//evaluate (current + offspring) Population
		currentList = evaluation.evaluate(currentList, problem);

		//Remove rule which is not used(winner = 0)
		String attributeId = (new NumberOfWinner<IntegerSolution>()).getAttributeId();

		currentList = currentList.stream()
						.map(x -> (MichiganSolution)x)
						.filter(x -> (int)x.getAttribute(attributeId) != 0)
						.collect(Collectors.toList());

		return currentList;
	}

}
