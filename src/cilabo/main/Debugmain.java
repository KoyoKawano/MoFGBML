package cilabo.main;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.uma.jmetal.solution.integersolution.IntegerSolution;

import cilabo.data.ClassLabel;
import cilabo.data.DataSet;
import cilabo.data.InputVector;
import cilabo.data.Pattern;
import cilabo.fuzzy.knowledge.Knowledge;
import cilabo.fuzzy.knowledge.factory.HomoTriangleKnowledgeFactory;
import cilabo.fuzzy.knowledge.membershipParams.HomoTriangle_2_3;
import cilabo.fuzzy.rule.Rule;
import cilabo.fuzzy.rule.antecedent.Antecedent;
import cilabo.fuzzy.rule.consequent.Consequent;
import cilabo.fuzzy.rule.consequent.RuleWeight;
import cilabo.gbml.solution.MichiganSolution;

public class Debugmain {

	public static void main(String[] args) {

		double[][] debugSet = {
				{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{0.1, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
				{0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
		};
		int[] classLabel0 = {0, 0, 0, 1};

 		ArrayList<ClassLabel> classLabel = new ArrayList<>();
 		for(int c : classLabel0) {
 			ClassLabel tmp = new ClassLabel();
 			tmp.addClassLabel(c);
 			classLabel.add(tmp);
 		}

 		ArrayList<InputVector> inputs = new ArrayList<>();
 		for(double[] p : debugSet) {
 			inputs.add(new InputVector(p));
 		}

 		DataSet train = new DataSet();
 		for(int i = 0; i < classLabel.size(); i++)
 			train.addPattern(new Pattern(i, inputs.get(i), classLabel.get(i)));
		train.setCnum(1);
		train.setNdim(8);

		Knowledge knowledge = HomoTriangleKnowledgeFactory.builder()
				.dimension(train.getNdim())
				.params(HomoTriangle_2_3.getParams())
				.build()
				.create();

		int[] antecedentIndex = {0, 0, 0, 0, 0, 0, 0, 0};
		Rule rule = new Rule(new Antecedent(knowledge, antecedentIndex),
							 new Consequent(new ClassLabel(),new RuleWeight()));

		int Size = 5000;
		List<Pair<Integer, Integer>> bounds = new ArrayList<>();
		for(int i = 0; i < Size * knowledge.getDimension(); i++) {
			bounds.add(Pair.of(0, knowledge.getFuzzySetNum(0)));
		}

		MichiganSolution solution = new MichiganSolution(bounds, 0, 0, rule);
		List<IntegerSolution> Pop = new ArrayList<>();
		for(int i = 0; i < Size; i++)
			Pop.add(solution.copy());

		//PittsburghSolution p_solution = new PittsburghSolution(bounds, 1, Pop, new SingleWinnerRuleSelection());

		System.out.println(rule.toString().getBytes().length);
/*************debug multi pattern heuristic rule generation**************
		double[][] debugSet = {
				{0.0, 0.0},
				{0.2, 0.2},
				{0.1, 0.4},
				{0.8, 0.5}
		};
		int[] classLabel0 = {0, 0, 0, 1};

 		ArrayList<ClassLabel> classLabel = new ArrayList<>();
 		for(int c : classLabel0) {
 			ClassLabel tmp = new ClassLabel();
 			tmp.addClassLabel(c);
 			classLabel.add(tmp);
 		}

 		ArrayList<InputVector> inputs = new ArrayList<>();
 		for(double[] p : debugSet) {
 			inputs.add(new InputVector(p));
 		}

 		DataSet train = new DataSet();
 		for(int i=0; i < classLabel.size(); i++)
 			train.addPattern(new Pattern(i, inputs.get(i), classLabel.get(i)));
		train.setCnum(1);
		train.setNdim(2);
 		int H = 3;

		Knowledge knowledge = HomoTriangleKnowledgeFactory.builder()
				.dimension(train.getNdim())
				.params(HomoTriangle_2_3.getParams())
				.build()
				.create();

  		MultipatternHeuristicRuleGeneration MPPRG = new MultipatternHeuristicRuleGeneration
  				(knowledge, H, train);

  		//System.out.print(knowledge.getFuzzySetNum(1));

  		//System.out.println(MPPRG.multipatternHeuristicRuleGeneration(train.getPattern(0)));


  		//test MichiganHeuristicVariation//
  		//population
  		//matingPopulation

  		//MichiganHeuristicVariation test = new MichiganHeuristicVariation(...)
  		//test.variate()
***********************************************************************************
*/
/*
		int rr = 0;
		int cc = 0;
		MichiganStyleFGBML(rr, cc);

	}

	public static void MichiganStyleFGBML(int rr, int cc) {
		String sep = File.separator;
		String trialRootDir = Consts.ALGORITHM_ID_DIR + sep + "trial" + String.valueOf(rr) + String.valueOf(cc);
		Output.mkdirs(trialRootDir);

		// Load "Pima" dataset
		String dataName = "dataset" + sep + "pima" + sep + "a"+rr+"_"+cc+"_pima-10tra.dat";
		DataSet train = new DataSet();
		Input.inputSingleLabelDataSet(train, dataName);

		// Parameters
		int populationSize = 30;
		int offspringPopulationSize = 6;
		// Problem
		int seed = 0;
		JMetalRandom.getInstance().setSeed(seed);
		ProblemMichiganFGBML<IntegerSolution> problem = new ProblemMichiganFGBML<>(seed, train);
		// Crossover
		double crossoverProbability = 1.0;
		CrossoverOperator<IntegerSolution> crossover = new UniformCrossover(crossoverProbability);
		// Mutation
		double mutationProbability = 1.0 / (double)train.getNdim();
		MutationOperator<IntegerSolution> mutation = new MichiganMutation(mutationProbability,
																	  problem.getKnowledge(),
																	  train);
//		int H = 2;
//		MultipatternHeuristicRuleGeneration ruleGenerateOperator = new MultipatternHeuristicRuleGeneration(problem.getKnowledge(),
//																								H,
//																								train);
		HeuristicRuleGeneration ruleGenerateOperator = new HeuristicRuleGeneration(problem.getKnowledge());
		// Termination
		int generation = 10000;
		int evaluations = populationSize + generation*offspringPopulationSize;
		int outputFrequency = 2000;
		Termination termination = new TerminationByEvaluations(evaluations);
		// Variation
		Variation<IntegerSolution> variation = new MichiganHeuristicVariation<IntegerSolution>(
													offspringPopulationSize, crossover, mutation,
													problem.getKnowledge(),
													problem.getConsequentFactory(),
													ruleGenerateOperator);
		// Replacement
		Replacement<IntegerSolution> replacement = new RuleAdditionAndNotUsedRuleRemoveStyleReplacement(problem);

		// Algorithm
		OnePlusOneESMichiganFGBML<IntegerSolution> algorithm
				= new OnePlusOneESMichiganFGBML<>(problem, populationSize, offspringPopulationSize,
									outputFrequency, trialRootDir,
									crossover, mutation, termination, variation, replacement);

		int observeFrequency = 1000;
		EvaluationObserver evaluationObserver = new EvaluationObserver(observeFrequency);
		algorithm.getObservable().register(evaluationObserver);

		algorithm.run();

		// Result
		List<RuleBasedClassifier> totalClassifiers = algorithm.getTotalClassifier();
		Metric metric = new ErrorRate();
		RuleBasedClassifier bestClassifier = null;
		double minValue = Double.MAX_VALUE;
		for(int i = 0; i < totalClassifiers.size(); i++) {
			double errorRate = (double)metric.metric(totalClassifiers.get(i), train);
			if(errorRate < minValue) {
				minValue = errorRate;
				bestClassifier = totalClassifiers.get(i);
			}
		}
		PostProcessing postProcessing = new RemoveNotBeWinnerProcessing(train);
		postProcessing.postProcess(bestClassifier);

		// Test data
		String testDataName = "dataset" + sep + "pima" + sep + "a"+rr+"_"+cc+"_pima-10tst.dat";
		DataSet test = new DataSet();
		Input.inputSingleLabelDataSet(test, testDataName);
		double errorRate = (double)metric.metric(bestClassifier, test);

		System.out.println();
		System.out.println("Error Rate(Train): " + minValue);
		System.out.println("Error Rate(Test) : " + errorRate);
		String fileName = trialRootDir + sep + "BestClassifier.txt";
		Output.writeln(fileName, bestClassifier.toString(), false);

		String str = String.valueOf(rr)+String.valueOf(cc);
		str += "," + minValue;
		str += "," + errorRate;
		Output.writeln(Consts.ALGORITHM_ID_DIR + sep + "errorRate.csv", str, true);
	}
*/
	}
}
