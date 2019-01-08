package org.deeplearning4j.solutions.feedforward.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Using MLP for classification from wine type analysis
 * There are 12 attributes, with label at last (0 - White, 1 - Red)
 *
 * wineData.csv will be used for training and assessment.
 *
 * Search for the part with *Enter your code here* and replace with model configuration
 *
 * [NOTE: Do not change other parts other than function getConfig(...)]
 */
public class MLPClassifierWine
{
    static int seed = 123;                      //Seed number for reproduction of the data
    static int batchSize = 100;
    static int numInputs = 12;                  //Number of attributes
    static int numClasses = 2;                  //Total number of labels
    static int epoch = 40;                      //Number of epochs
    static double splitRatio = 0.8;             //Splitting of data into training and testing data set
    static final int MINEXAMPLESPERCLASS = 1599;//The minimum amount of data per file
    static double learningRate = 0.01;          //How fast to adjust weights to minimize error

    public static void main(String[] args) throws Exception
    {
        //Set file
        File dataFile = new ClassPathResource("/classification/wine/wineData.csv").getFile();

        //Read from file
        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader rr = new CSVRecordReader(numLinesToSkip, delimiter);
        rr.initialize(new FileSplit(dataFile));

        //Get all data before shuffle
        DataSetIterator bufferIter = new RecordReaderDataSetIterator(rr, MINEXAMPLESPERCLASS * 2, numInputs, numClasses);

        List<DataSet> bufferList = bufferIter.next().asList();

        //Shuffle data so that every batch has a mixed of different labels
        Collections.shuffle(bufferList, new Random(seed));

        //Split training and testing dataIter
        int trainEndIndex = (int) (Math.ceil(bufferList.size() * splitRatio));
        List<DataSet> trainingList = bufferList.subList(0, trainEndIndex);
        List<DataSet> testingList = bufferList.subList(trainEndIndex, bufferList.size());

        DataSetIterator trainIter = new ListDataSetIterator(trainingList, batchSize);
        DataSetIterator testIter = new ListDataSetIterator(testingList, batchSize);

        //Get network configuration
        MultiLayerConfiguration config = getConfig(numInputs, numClasses, learningRate);

        if(config == null)
        {
            System.out.println("Configuration not set right. Abort");
            return;
        }

        //Define network
        MultiLayerNetwork network = new MultiLayerNetwork(config);

        network.setListeners(new ScoreIterationListener(10));

        //Starts training
        for(int i = 0; i < epoch; ++i)
        {
            network.fit(trainIter);
        }

        //Evaluation
        Evaluation eval = network.evaluate(testIter);

        System.out.println(eval.stats());
    }

    /**
     * Build network configuration
     *
     * @param numInputs  input layer nodes
     * @param numOutputs output layer nodes
     * @return MultiLayerConfiguration with network configuration
     */
    public static MultiLayerConfiguration getConfig(int numInputs, int numOutputs, double learningRate)
    {

        /**
         * Enter your code here
         */

        return null; //change to return MultiLayerConfiguration instance
    }
}
