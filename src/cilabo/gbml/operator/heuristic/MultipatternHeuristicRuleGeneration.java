//written by koyo kawano
package cilabo.gbml.operator.heuristic;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.uma.jmetal.operator.Operator;

import cilabo.data.ClassLabel;
import cilabo.data.DataSet;
import cilabo.data.Pattern;
import cilabo.fuzzy.knowledge.Knowledge;
import cilabo.fuzzy.rule.antecedent.Antecedent;
import cilabo.utility.GeneralFunctions;
import cilabo.utility.Random;

public class MultipatternHeuristicRuleGeneration implements Operator<List<Pattern>, List<Antecedent>>, PatternBaseRuleGeneration{
	Knowledge knowledge;
	// H is # of patterns using for rule generation
	// 1 pattern is base pattern, (H-1) patterns is support pattern
	// train is taraining data
	int H;

	ArrayList<Pattern> train;

	public MultipatternHeuristicRuleGeneration(Knowledge knowledge, int H, DataSet Train) {
		this.knowledge = knowledge;
		this.H = H;
		this.train = Train.getPatterns();
	}

	@Override
	public List<Antecedent> execute(List<Pattern> erroredPatterns){
		List<Antecedent> generatedRules = new ArrayList<>();
		for(int i = 0; i < erroredPatterns.size(); i++) {
			generatedRules.add(multipatternHeuristicRuleGeneration(erroredPatterns.get(i)));
		}
		return generatedRules;
	}

	/*
	複数パターンによるルール生成
	@param : basepattern : 識別器の誤識別パターンから選択

	1. basepatternからサポートパターンを(H-1)個選択
	   選び方は，basepatternとクラスラベルが一致するパターンからランダム
	2. ルーレットに使用する確率の計算
	3. ルーレット選択

	注意：通常のヒューリスティックルール生成との違いは，ルーレット選択後にDon't careを適用しない
	      ルーレットに使用する確率は通常の生成法とは異なり，サポートパターンとベースパターンを両方
	      カバーするファジィ集合がない場合のみ確率0(Don't care)が適用される．
	*/

	/*
	以下まだ未完成
	TODO
	basepattern からクラスラベルが一致するsupportpatternをランダムに(H-1)個選択
	ルーレットに使う確率の計算
	ルーレット選択
	 */
	public Antecedent multipatternHeuristicRuleGeneration(Pattern basePattern){
		/** Number of attribute. */
		int dimension = basePattern.getInputVector().getVector().length;

		/* Select fuzzy sets */
		int[] antecedentIndex = new int[dimension];
		for(int n = 0; n < dimension; n++) {
			//Categorical Dimension
			if(basePattern.getDimValue(n) < 0) {
				antecedentIndex[n] = (int)basePattern.getDimValue(n);
			}
			//Numerical Dimension
			else {
				//サポートパターン選択
				List<Pattern> baseSupportPattern = SelectSupportPattern(basePattern);
				baseSupportPattern.add(basePattern);

				double[] membershipValueRoulette = new double[knowledge.getFuzzySetNum(n)];
				double sumMembershipValue = 0.0;

				// Make roulette
				// don't care value is equal to zero
				membershipValueRoulette[0] = 0.0;
				// 全てのファジィ集合Aに対して計算
				for(int f = 1; f < knowledge.getFuzzySetNum(n); f++) {
					//全てのルール生成に用いるパターン(base pattern + support pattern)に対して計算
					final int _n = n;
					final int _f = f;
					double membershipValue = baseSupportPattern
											.stream()
											.map(x -> knowledge.getMembershipValue(x.getDimValue(_n), _n, _f))
											.min(Comparator.naturalOrder()).orElse(-1.0);

					//ファジィ集合Auのメンバシップ値の最小値が0.5以下であれば0.0
					if(membershipValue <= 0.5) {
						membershipValue = 0.0;
					}

					sumMembershipValue += membershipValue;
					membershipValueRoulette[f] = sumMembershipValue;
				}
				//全てのルーレットの確率が0の時，Don't care
				if(sumMembershipValue == 0.0){
					antecedentIndex[n] = 0;
				}
				else {
				// Select fuzzy set
					double arrow = Random.getInstance().getGEN().nextDouble() * sumMembershipValue;
					for(int f = 0; f < knowledge.getFuzzySetNum(n); f++) {
						if(arrow < membershipValueRoulette[f]) {
							antecedentIndex[n] = f;
							break;
						}
					}
				}
			}
		}

		return Antecedent.builder()
							.knowledge(knowledge)
							.antecedentIndex(antecedentIndex)
							.build();
	}

	/*
	 basepatternとクラスラベルが等しいパターンをランダムに(H-1)個trainから選択する．
	 @param basePattern : base pattern
	 @param train : training data
	 */
	public List<Pattern> SelectSupportPattern(Pattern basePattern){
		ClassLabel baseClass = basePattern.getTrueClass();

		int candidateNum = (int)train.stream()
							.filter(x-> x.getTrueClass().getClassLabel() == baseClass.getClassLabel())
							.count();

		Integer[] randIndexlist = GeneralFunctions.samplingWithout((candidateNum), (candidateNum), Random.getInstance().getGEN());
		int baseID = basePattern.getID();
		List<Pattern> supportPattern = new ArrayList<>();
		for(int i = 0; i < (H - 1); i++) {

			if(train.get(randIndexlist[i]).getID() != baseID) {
				supportPattern.add(train.get(randIndexlist[i]));
			}
		}
		return supportPattern;
	}

	public Knowledge getKnowledge() {
		return this.knowledge;
	}

	@Override
	public Antecedent ruleGenerate(Pattern pattern) {
		return multipatternHeuristicRuleGeneration(pattern);
	}
}

