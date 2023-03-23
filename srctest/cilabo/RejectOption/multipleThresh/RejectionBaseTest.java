package cilabo.RejectOption.multipleThresh;

import static org.junit.Assert.*;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;

import cilabo.data.DataSet;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.fuzzy.classifier.operator.classification.factory.SingleWinnerRuleSelection;
import cilabo.fuzzy.knowledge.Knowledge;
import cilabo.fuzzy.knowledge.factory.HomoTriangleKnowledgeFactory;
import cilabo.fuzzy.knowledge.membershipParams.HomoTriangle_2_3;
import cilabo.fuzzy.rule.Rule;
import cilabo.fuzzy.rule.antecedent.Antecedent;
import cilabo.fuzzy.rule.consequent.Consequent;
import cilabo.fuzzy.rule.consequent.ConsequentFactory;
import cilabo.fuzzy.rule.consequent.factory.MoFGBML_Learning;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.ClassificationDataInfo;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.EstimateThreshold;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.RejectionBase;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.SingleThreshold;
import cilabo.utility.Input;

public class RejectionBaseTest {

	@Test
	public void multipleThresholdTest() {

		String sep = File.separator;
		String dataName = "dataset" + sep + "cilabo" + sep + "kadai5_pattern1.txt";
		DataSet train = new DataSet();
		Input.inputSingleLabelDataSet(train, dataName);

		Knowledge knowledge = HomoTriangleKnowledgeFactory.builder()
				.dimension(train.getNdim())
				.params(HomoTriangle_2_3.getParams())
				.build()
				.create();

		Antecedent[] antecedents = new Antecedent[5];
		antecedents[0] = Antecedent.builder()
				.knowledge(knowledge)
				.antecedentIndex(new int[] {0, 5})
				.build();
		antecedents[1] = Antecedent.builder()
				.knowledge(knowledge)
				.antecedentIndex(new int[] {3, 4})
				.build();
		antecedents[2] = Antecedent.builder()
				.knowledge(knowledge)
				.antecedentIndex(new int[] {4, 3})
				.build();
		antecedents[3] = Antecedent.builder()
				.knowledge(knowledge)
				.antecedentIndex(new int[] {4, 4})
				.build();
		antecedents[4] = Antecedent.builder()
				.knowledge(knowledge)
				.antecedentIndex(new int[] {5, 0})
				.build();

		ConsequentFactory consequentFactory = new MoFGBML_Learning(train);

		List<Rule> ruleSet = new ArrayList<Rule>()
;
		for(Antecedent antecedent : antecedents) {

			Consequent consequent = consequentFactory.learning(antecedent);

			Rule rule = Rule.builder()
					.antecedent(antecedent)
					.consequent(consequent)
					.build();

			ruleSet.add(rule);
		}

		RuleBasedClassifier ruleBasedClassifier = new RuleBasedClassifier();

		ruleBasedClassifier.setClassification(new SingleWinnerRuleSelection());

		for(Rule rule : ruleSet) {
			ruleBasedClassifier.addRule(rule);
		}

		int kmax = 500;
		double deltaT = 0.001;
		double Rmax = 0.5;

		List<ClassificationDataInfo> trainInfo = train.getPatterns().stream()
				.map(x-> new ClassificationDataInfo(x, ruleBasedClassifier))
				.collect(Collectors.toList());

		RejectionBase single = new SingleThreshold();
		EstimateThreshold estimateThreshold = new EstimateThreshold(trainInfo, single, kmax, deltaT, Rmax);

		double[] expected = estimateThreshold.run();

		double diff = 0.0001;

		assertEquals(expected[0], 1.0, diff);
		assertEquals(expected[1], 0.116666666666666, diff);
	}
}
