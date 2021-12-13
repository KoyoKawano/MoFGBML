package cilabo.main;

import java.io.File;
import java.util.List;

import org.uma.jmetal.component.replacement.Replacement;
import org.uma.jmetal.component.termination.Termination;
import org.uma.jmetal.component.termination.impl.TerminationByEvaluations;
import org.uma.jmetal.component.variation.Variation;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.solution.integersolution.IntegerSolution;
import org.uma.jmetal.util.observer.impl.EvaluationObserver;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

import cilabo.data.DataSet;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.fuzzy.classifier.operator.postProcessing.PostProcessing;
import cilabo.fuzzy.classifier.operator.postProcessing.factory.RemoveNotBeWinnerProcessing;
import cilabo.gbml.algorithm.MichiganFGBML;
import cilabo.gbml.component.replacement.SingleObjectiveMaximizeReplacementWithoutOffspringFitness;
import cilabo.gbml.component.variation.MichiganHeuristicVariation;
import cilabo.gbml.operator.crossover.UniformCrossover;
import cilabo.gbml.operator.heuristic.MultipatternHeuristicRuleGeneration;
import cilabo.gbml.operator.mutation.MichiganMutation;
import cilabo.gbml.problem.impl.michigan.ProblemMichiganFGBML;
import cilabo.metric.ErrorRate;
import cilabo.metric.Metric;
import cilabo.utility.Input;
import cilabo.utility.Output;

public class Debugmain {

	public static void main(String[] args) {

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
		int populationSize = 60;
		int offspringPopulationSize = 12;
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
		int H = 2;
		MultipatternHeuristicRuleGeneration ruleGenerateOperator = new MultipatternHeuristicRuleGeneration(problem.getKnowledge(),
																								H,
																								train);
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
		Replacement<IntegerSolution> replacement = new SingleObjectiveMaximizeReplacementWithoutOffspringFitness<>();

		// Algorithm
		MichiganFGBML<IntegerSolution> algorithm
				= new MichiganFGBML<>(problem, populationSize, offspringPopulationSize,
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
}
