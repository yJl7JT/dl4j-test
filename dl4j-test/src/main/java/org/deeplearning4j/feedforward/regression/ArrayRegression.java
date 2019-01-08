package org.deeplearning4j.feedforward.regression;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Using MLP to predict 4 output values from 2 input values
 * {0.4, 0.5, 0.8} -> {0.23, 0.855, 1.6, 2.72}
 *
 * Search for the part with *Enter your code here* and replace with model configuration
 *
 * [NOTE: Do not change other parts other than function getConfig(...)]
 */
public class ArrayRegression
{
    private static Logger log = LoggerFactory.getLogger(ArrayRegression.class);

    public static void main(String[] args) throws Exception
    {
        //Declare the input and output data in INDArray format
        double[] inputArr = new double[]{0.4, 0.5, 0.8};
        double[] outputArr = new double[]{0.23, 0.855, 1.6, 2.72};

        INDArray input = Nd4j.create(inputArr);
        INDArray output = Nd4j.create(outputArr);

        //Set up the network configuration
        MultiLayerConfiguration config = getConfig(input.length(), output.length());// number of input nodes, number of output nodes

        if(config == null)
        {
            System.out.println("Null network configuration. End of program.");
            return;
        }

        //Declare MultiLayerNetwork, train the network
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(5));

        int epochs = 50;           // Number of epochs(full passes of the data)
        int evaluationInterval = 5; // Evaluate on a designated interval

        for (int i = 0; i < epochs; ++i)
        {
            model.fit(input, output);

            if((i % evaluationInterval) == 0)
            {
                INDArray predicted = model.output(input);
                log.info("Predicted value: " + predicted.toString());

            }
            Thread.sleep(100);
        }
    }

    /**
     * Build network configuration
     *
     * @param numInputs  input layer nodes
     * @param numOutputs output layer nodes
     * @return MultiLayerConfiguration with network configuration
     */
    public static MultiLayerConfiguration getConfig(long numInputs, long numOutputs) {

        /**
         * Enter your code here
         */

        return null; //change to return MultiLayerConfiguration instance
    }

}
