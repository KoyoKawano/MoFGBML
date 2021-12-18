package cilabo.labo.main.kawano.RuleAdditionMichiganStyleFGBML;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Date;
import java.util.List;

import org.uma.jmetal.component.replacement.Replacement;
import org.uma.jmetal.component.termination.Termination;
import org.uma.jmetal.component.termination.impl.TerminationByEvaluations;
import org.uma.jmetal.component.variation.Variation;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.solution.integersolution.IntegerSolution;
import org.uma.jmetal.util.JMetalException;
import org.uma.jmetal.util.observer.impl.EvaluationObserver;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

import cilabo.data.DataSet;
import cilabo.data.impl.TrainTestDatasetManager;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.gbml.algorithm.OnePlusOneESMichiganFGBML;
import cilabo.gbml.component.replacement.RuleAdditionAndNotUsedRuleRemoveStyleReplacement;
import cilabo.gbml.component.variation.MichiganHeuristicVariation;
import cilabo.gbml.operator.crossover.UniformCrossover;
import cilabo.gbml.operator.heuristic.HeuristicRuleGeneration;
import cilabo.gbml.operator.mutation.MichiganMutation;
import cilabo.gbml.problem.impl.michigan.ProblemMichiganFGBML;
import cilabo.main.Consts;
import cilabo.metric.ErrorRate;
import cilabo.metric.Metric;
import cilabo.utility.Output;
import cilabo.utility.Parallel;
import cilabo.utility.Random;

public class RuleAdditionMichiganStyleFGBML_main {
	public static void main(String args[]) throws JMetalException, FileNotFoundException {
		String sep = File.separator;

		/* ********************************************************* */
		System.out.println();
		System.out.println("==== INFORMATION ====");
		String version = "1.0";
		System.out.println("main: " + RuleAdditionMichiganStyleFGBML_main.class.getCanonicalName());
		System.out.println("version: " + version);
		System.out.println();
		System.out.println("Algorithm: Rule Addition Michigan Style Fuzzy Genetics-Based Machine Learning");
		System.out.println();
		/* ********************************************************* */
		// Load consts.properties
		Consts.set("consts");
		// make result directory
		Output.mkdirs(Consts.ROOTFOLDER);

		// set command arguments to static variables
		CommandLineArgs.loadArgs(CommandLineArgs.class.getCanonicalName(), args);
		// Output constant parameters
		String fileName = Consts.EXPERIMENT_ID_DIR + sep + "Consts.txt";
		Output.writeln(fileName, Consts.getString(), true);
		Output.writeln(fileName, CommandLineArgs.getParamsString(), true);

		// Initialize ForkJoinPool
		Parallel.getInstance().initLearningForkJoinPool(CommandLineArgs.parallelCores);

		System.out.println("Processors: " + Runtime.getRuntime().availableProcessors() + " ");
		System.out.print("args: ");
		for(int i = 0; i < args.length; i++) {
			System.out.print(args[i] + " ");
		}
		System.out.println();
		System.out.println("=====================");
		System.out.println();

		/* ********************************************************* */
		System.out.println("==== EXPERIMENT =====");
		Date start = new Date();
		System.out.println("START: " + start);

		/* Random Number ======================= */
		Random.getInstance().initRandom(Consts.RAND_SEED);
		JMetalRandom.getInstance().setSeed(Consts.RAND_SEED);

		/* Load Dataset ======================== */
		TrainTestDatasetManager datasetManager = new TrainTestDatasetManager();
		datasetManager.loadTrainTestFiles(CommandLineArgs.trainFile, CommandLineArgs.testFile);

		/* Run MoFGBML algorithm =============== */
		DataSet train = datasetManager.getTrains().get(0);
		DataSet test = datasetManager.getTests().get(0);
		RuleAdditionMichiganStyleFGBML(train, test, CommandLineArgs.H);
		/* ===================================== */

		Date end = new Date();
		System.out.println("END: " + end);
		System.out.println("=====================");
		/* ********************************************************* */

		System.exit(0);
	}

		public static void RuleAdditionMichiganStyleFGBML(DataSet train, DataSet test, int H) {
			String sep = File.separator;

			// Parameters
			int populationSize = Consts.initialPopulationSize;
			int offspringPopulationSize = Consts.offspringPopulationSize;
			// Problem
			int seed = Consts.RAND_SEED;
			JMetalRandom.getInstance().setSeed(seed);
			ProblemMichiganFGBML<IntegerSolution> problem = new ProblemMichiganFGBML<>(seed, train);
			// Crossover
			double crossoverProbability = Consts.MICHIGAN_CROSS_RT;
			CrossoverOperator<IntegerSolution> crossover = new UniformCrossover(crossoverProbability);
			// Mutation
			double mutationProbability = 1.0 / (double)train.getNdim();
			MutationOperator<IntegerSolution> mutation = new MichiganMutation(mutationProbability,
																		  problem.getKnowledge(),
																		  train);
//			int H = 2;
//			MultipatternHeuristicRuleGeneration ruleGenerateOperator = new MultipatternHeuristicRuleGeneration(problem.getKnowledge(),
//																									H,
//																									train);
			HeuristicRuleGeneration ruleGenerateOperator = new HeuristicRuleGeneration(problem.getKnowledge());
			// Termination
			Termination termination = new TerminationByEvaluations(Consts.initialPopulationSize + Consts.terminateGeneration * Consts.offspringPopulationSize);
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
					= new OnePlusOneESMichiganFGBML<>(problem,
													  populationSize,
													  offspringPopulationSize,
													  Consts.outputFrequency,
													  Consts.EXPERIMENT_ID_DIR,
													  crossover,
													  mutation,
													  termination,
													  variation,
													  replacement
													  );

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

			// Test data
			double errorRate = (double)metric.metric(bestClassifier, test);
			int ruleNum = bestClassifier.getRuleNum();
			System.out.println();
			System.out.println("Error Rate(Train): " + minValue);
			System.out.println("Error Rate(Test) : " + errorRate);
			String fileName = Consts.EXPERIMENT_ID_DIR + sep + "BestClassifier.txt";
			Output.writeln(fileName, bestClassifier.toString(), false);

			String str = CommandLineArgs.experimentID;
			str += "," + minValue;
			str += "," + errorRate;
			str += "," + ruleNum;
			Output.writeln(Consts.ALGORITHM_ID_DIR + sep + "errorRate.csv", str, true);
		}

}
