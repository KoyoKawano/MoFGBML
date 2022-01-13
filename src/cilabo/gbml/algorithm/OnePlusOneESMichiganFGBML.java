package cilabo.gbml.algorithm;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.uma.jmetal.algorithm.impl.AbstractEvolutionaryAlgorithm;
import org.uma.jmetal.component.evaluation.Evaluation;
import org.uma.jmetal.component.initialsolutioncreation.InitialSolutionsCreation;
import org.uma.jmetal.component.initialsolutioncreation.impl.RandomSolutionsCreation;
import org.uma.jmetal.component.replacement.Replacement;
import org.uma.jmetal.component.selection.MatingPoolSelection;
import org.uma.jmetal.component.selection.impl.NaryTournamentMatingPoolSelection;
import org.uma.jmetal.component.termination.Termination;
import org.uma.jmetal.component.variation.Variation;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.operator.selection.SelectionOperator;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.util.JMetalLogger;
import org.uma.jmetal.util.SolutionListUtils;
import org.uma.jmetal.util.comparator.ObjectiveComparator;
import org.uma.jmetal.util.comparator.ObjectiveComparator.Ordering;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;
import org.uma.jmetal.util.observable.Observable;
import org.uma.jmetal.util.observable.ObservableEntity;
import org.uma.jmetal.util.observable.impl.DefaultObservable;

import cilabo.data.DataSet;
import cilabo.fuzzy.classifier.RuleBasedClassifier;
import cilabo.gbml.component.evaluation.MichiganEvaluation;
import cilabo.gbml.problem.AbstractMichiganGBML_Problem;
import cilabo.gbml.problem.impl.michigan.ProblemMichiganFGBML;
import cilabo.gbml.solution.MichiganSolution;
import cilabo.metric.ErrorRate;
import cilabo.metric.Metric;
import cilabo.util.fileoutput.MichiganSolutionListOutput;
import cilabo.utility.Output;

public class OnePlusOneESMichiganFGBML<S extends Solution<?>> extends AbstractEvolutionaryAlgorithm<S, List<S>>
							implements ObservableEntity
{
	private int evaluations;
	private int populationSize;
	private int offspringPopulationSize;
	private int frequency;
	private String outputRootDir;

	protected SelectionOperator<List<S>, S> selectionOperator;
	protected CrossoverOperator<S> crossoverOperator;
	protected MutationOperator<S> mutationOperator;

	private Map<String, Object> algorithmStatusData;

	private InitialSolutionsCreation<S> initialSolutionsCreation;
	private Termination termination;
	private Evaluation<S> evaluation ;
	private Replacement<S> replacement;
	private Variation<S> variation;
	private MatingPoolSelection<S> selection;

	private long startTime;
	private long totalComputingTime;

	private double minErrorRate;

	private Observable<Map<String, Object>> observable;

	private List<RuleBasedClassifier> totalClassifiers = new ArrayList<>();

	/** Constructor */
	public OnePlusOneESMichiganFGBML(
			Problem<S> problem,
			int populationSize,
			int offspringPopulationSize,
			int frequency,
			String outputRootDir,
			CrossoverOperator<S> crossoverOperator,
			MutationOperator<S> mutationOperator,
			Termination termination,
			Variation<S> variation,
			Replacement<S> replacement)
	{
		this.populationSize = populationSize;
		this.offspringPopulationSize = offspringPopulationSize;
		this.frequency = frequency;
		this.outputRootDir = outputRootDir;

		this.problem = problem;

		this.crossoverOperator = crossoverOperator;
		this.mutationOperator = mutationOperator;
		this.termination = termination;
		this.variation = variation;
		this.replacement = replacement;

		this.initialSolutionsCreation = new RandomSolutionsCreation<>(problem, populationSize);
		this.evaluation = new MichiganEvaluation<S>();

		this.selection =
				new NaryTournamentMatingPoolSelection<>(
						2,	//Tournament Size
						variation.getMatingPoolSize(),
						new ObjectiveComparator<>(0, Ordering.DESCENDING));	//Single Objective, Maximize

		this.algorithmStatusData = new HashMap<>();
		this.observable = new DefaultObservable<>("Michigan-type FGBML with Single-objective");
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	@Override
	public void run() {
		startTime = System.currentTimeMillis();

		/*** START ***/
	    List<S> offspringPopulation;
	    List<S> matingPopulation;


	    population = createInitialPopulation();
	    population = evaluatePopulation(population);
		List<S> bestPopulation = population.stream()
											.map(x-> (S)(MichiganSolution)x.copy())
											.collect(Collectors.toList());

		DataSet train = ((ProblemMichiganFGBML)problem).getEvaluationDataset();
		Metric metric = new ErrorRate();
		minErrorRate = (double)metric
							.metric(((AbstractMichiganGBML_Problem<S>)problem)
							.population2classifier(population), train);

	    initProgress();
	    while (!isStoppingConditionReached()) {
	      matingPopulation = selection(population);
	      offspringPopulation = reproduction(matingPopulation);
	      population = replacement(population, offspringPopulation);
	      population = evaluatePopulation(population);

	      double errorRate = (double)metric
								.metric(((AbstractMichiganGBML_Problem<S>)problem)
								.population2classifier(population), train);

	      if(errorRate < minErrorRate) {
	    	  minErrorRate = errorRate;
	    	  bestPopulation = population.stream()
								.map(x-> (S)(MichiganSolution)x.copy())
								.collect(Collectors.toList());
	      }
	      else {
	    	  population = bestPopulation.stream()
							.map(x-> (S)(MichiganSolution)x.copy())
							.collect(Collectors.toList());

	      }
	      populationSize = population.size();
	      updateProgress();
	    }
		/***  END ***/
		totalComputingTime = System.currentTimeMillis() - startTime;
	}

	@Override
	protected void initProgress() {
		evaluations = populationSize;

	    algorithmStatusData.put("EVALUATIONS", evaluations);
	    algorithmStatusData.put("POPULATION", population);
	    algorithmStatusData.put("COMPUTING_TIME", System.currentTimeMillis() - startTime);

	    observable.setChanged();
	    observable.notifyObservers(algorithmStatusData);
	}

	@SuppressWarnings("unchecked")
	@Override
	protected void updateProgress() {
		this.totalClassifiers.add(((AbstractMichiganGBML_Problem<S>)problem).population2classifier(population));
		evaluations += offspringPopulationSize;
		algorithmStatusData.put("EVALUATIONS", evaluations);
		algorithmStatusData.put("POPULATION", population);
		algorithmStatusData.put("COMPUTING_TIME", System.currentTimeMillis() - startTime);

		observable.setChanged();
		observable.notifyObservers(algorithmStatusData);

		String sep = File.separator;
		Integer evaluations = (Integer)algorithmStatusData.get("EVALUATIONS");
		if (evaluations!=null) {
			if (evaluations % frequency == 0) {
				new MichiganSolutionListOutput(getPopulation())
					.printMichiganSolutionFormatsToCSV(new DefaultFileOutputContext(outputRootDir+sep+"solutions-"+evaluations+".csv"), getPopulation());

				String str = Double.toString(minErrorRate) + "," + Integer.toString(populationSize);
				Output.writeln(outputRootDir+sep+"errorAndRuleNum.csv", str, true);
			}
		}
		else {
			JMetalLogger.logger.warning(getClass().getName()
			+ ": The algorithm has not registered yet any info related to the EVALUATIONS key");
		}

	}

	@Override
	protected boolean isStoppingConditionReached() {
		return termination.isMet(algorithmStatusData);
	}

	@Override
	protected List<S> createInitialPopulation() {
		return initialSolutionsCreation.create();
	}

	@Override
	protected List<S> evaluatePopulation(List<S> population) {
		return  evaluation.evaluate(population, getProblem());
	}

	@Override
	protected List<S> selection(List<S> population) {
		return this.selection.select(population);
	}

	@Override
	protected List<S> reproduction(List<S> matingPool){
		return variation.variate(population, matingPool);
	}

	@Override
	protected List<S> replacement(List<S> population, List<S> offspringPopulation) {
		return replacement.replace(population, offspringPopulation);
	}

	@Override
	public List<S> getResult(){
		return SolutionListUtils.getNonDominatedSolutions(getPopulation());
	}

	@Override
	public String getName() {
		return "Michigan FGBML with Single-objective";
	}

	@Override
	public String getDescription() {
		return "Single-objective Michigan-type Fuzzy Genetics-Based Machine Learning";
	}

	public Map<String, Object> getAlgorithmStatusData() {
		return algorithmStatusData;
	}

	@Override
	public Observable<Map<String, Object>> getObservable() {
		return observable;
	}

	public long getTotalComputingTime() {
		return totalComputingTime;
	}

	public long getEvaluations() {
		return evaluations;
	}

	public List<RuleBasedClassifier> getTotalClassifier() {
		return this.totalClassifiers;
	}

	/* Setter */

	public OnePlusOneESMichiganFGBML<S> setSelectionOperator(SelectionOperator<List<S>, S> selectionOperator) {
		this.selectionOperator = selectionOperator;
		return this;
	}

	public OnePlusOneESMichiganFGBML<S> setCrossoverOperator(CrossoverOperator<S> crossoverOperator) {
		this.crossoverOperator = crossoverOperator;
		return this;
	}

	public OnePlusOneESMichiganFGBML<S> setMutationOperator(MutationOperator<S> mutationOperator) {
		this.mutationOperator = mutationOperator;
		return this;
	}

	public OnePlusOneESMichiganFGBML<S> setInitialSolutionsCreation(InitialSolutionsCreation<S> initialSolutionsCreation) {
		this.initialSolutionsCreation = initialSolutionsCreation;
		return this;
	}

	public OnePlusOneESMichiganFGBML<S> setTermination(Termination termination) {
		this.termination = termination;
		return this;
	}

	public OnePlusOneESMichiganFGBML<S> setEvaluation(Evaluation<S> evaluation) {
		this.evaluation = evaluation;
		return this;
	}

	public OnePlusOneESMichiganFGBML<S> setReplacement(Replacement<S> replacement) {
		this.replacement = replacement;
		return this;
	}

	public OnePlusOneESMichiganFGBML<S> setVariation(Variation<S> variation) {
		this.variation = variation;
		return this;
	}

	public OnePlusOneESMichiganFGBML<S> setSelection(MatingPoolSelection<S> selection) {
		this.selection = selection;
		return this;
	}
}
