package cilabo.main;

import java.util.ArrayList;

import cilabo.data.ClassLabel;
import cilabo.data.DataSet;
import cilabo.data.InputVector;
import cilabo.data.Pattern;
import cilabo.fuzzy.knowledge.Knowledge;
import cilabo.fuzzy.knowledge.factory.HomoTriangleKnowledgeFactory;
import cilabo.fuzzy.knowledge.membershipParams.HomoTriangle_2_3;
import cilabo.gbml.operator.heuristic.MultipatternHeuristicRuleGeneration;

public class Debugmain {

	public static void main(String[] args) {

		double[][] debugSet = {
				{0.0, 0.0},
				{0.2, 0.2},
				{0.1, 0.4}
		};
		int[] classLabel0 = {0, 0, 0};

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
 		ArrayList<Pattern> pattern = new ArrayList<>();
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

  		System.out.println(MPPRG.multipatternHeuristicRuleGeneration(train.getPattern(0)));

	}

}
