package cilabo.labo.main.kawano.tryCode;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

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
import cilabo.labo.main.kawano.rejectOption.TwoStageRejectOption.KNN;
import cilabo.labo.main.kawano.rejectOption.TwoStageRejectOption.SecondClassifier;
import cilabo.labo.main.kawano.rejectOption.TwoStageRejectOption.TwoStageRejectOption;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.ClassWiseThresholds;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.ClassificationDataInfo;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.EstimateThreshold;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.RejectOptionMetric;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.RejectionBase;
import cilabo.labo.main.kawano.rejectOption.multipleThresh.SingleThreshold;
import cilabo.utility.Input;

public class Try {

	public static void main(String[] args) {

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


//		Antecedent[] antecedents = new Antecedent[3];
//		antecedents[0] = Antecedent.builder()
//				.knowledge(knowledge)
//				.antecedentIndex(new int[] {0, 3})
//				.build();
//		antecedents[1] = Antecedent.builder()
//				.knowledge(knowledge)
//				.antecedentIndex(new int[] {0, 5})
//				.build();
//
//		antecedents[2] = Antecedent.builder()
//				.knowledge(knowledge)
//				.antecedentIndex(new int[] {5, 0})
//				.build();

		ConsequentFactory consequentFactory = new MoFGBML_Learning(train);

		List<Rule> ruleSet = new ArrayList<Rule>();

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
		double Rmax = 0.50;

		List<ClassificationDataInfo> trainInfo = train.getPatterns().stream()
				.map(x-> new ClassificationDataInfo(x, ruleBasedClassifier))
				.collect(Collectors.toList());


		SingleThreshold single = new SingleThreshold();

		EstimateThreshold estimateThreshold = new EstimateThreshold(trainInfo, single, kmax, deltaT, Rmax);

		double[] expected = estimateThreshold.run();

		System.out.println(expected[0]);
		System.out.println(expected[1]);

		RejectionBase cwt = new ClassWiseThresholds(train);

		EstimateThreshold estimateClassWiseThresholds = new EstimateThreshold(trainInfo, cwt, kmax, deltaT, Rmax);

		expected = estimateClassWiseThresholds.run();

		double[] T = estimateClassWiseThresholds.getThreshold();

		DoubleStream.of(T).forEach(System.out::println);



		SecondClassifier secondClassifier = new KNN(train, 3);
		TwoStageRejectOption twoStageRejectOption = new TwoStageRejectOption(cwt, secondClassifier);


		RejectOptionMetric metric = new RejectOptionMetric(twoStageRejectOption);
		System.out.println(metric.culcAcc(trainInfo, estimateClassWiseThresholds.getThreshold()));
		System.out.println(metric.culcRejectRate(trainInfo, estimateClassWiseThresholds.getThreshold()));

	}

}
