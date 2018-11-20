package cmps142_hw4;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class LogisticRegression_withBias {

        /** the learning rate */
        private double rate=0.01;

        /** the weights to learn */
        private double[] weights;

        /** the number of iterations */
        private int ITERATIONS = 200;

        /** TODO: Constructor initializes the weight vector. Initialize it by setting it to the 0 vector. **/
        public LogisticRegression_withBias(int n) { // n is the number of weights to be learned
            weights = new double[n+1];
            for(double w: weights)
                w = 0.0;

            // zeroth index is our bias
        }

        /** TODO: Implement the function that returns the L2 norm of the weight vector **/
        private double weightsL2Norm(){
            // L2 Norm = sqrt(summation(weights^2))

            // summation of the weights
            double sum = 0;
            for(int i = 1; i<weights.length; i++)
                sum += Math.pow(weights[i],2);

            // sqrt of the weights
            return Math.sqrt(sum);
        }

        /** TODO: Implement the sigmoid function **/
        private static double sigmoid(double z) {
            return 1.0/(1.0+Math.pow(10,-z));
        }

        /** TODO: Helper function for prediction **/
        /** Takes a test instance as input and outputs the probability of the label being 1 **/
        /** This function should call sigmoid() **/
        private double probPred1(double[] x) {
            // Logistic Regression 1: slide 15, 16

            // calculates dot product of weights and x
            /** TODO: Check inconsistency with Slide 16, where w0 is added independently, but in slide 18 w0 is included as a part of the summation**/
            double dotProduct = weights[0];
            for(int i = 0; i<x.length; i++){       // bias = weights[0]
                dotProduct += weights[i+1] * x[i];
            }

            return sigmoid(dotProduct);
        }

        /** TODO: The prediction function **/
        /** Takes a test instance as input and outputs the predicted label **/
        /** This function should call probPred1() **/
        public int predict(double[] x) {
            // Logistic Regression 1: slide 20
            // if P(Y=0|X)>(Y=1|X), then classify as 0
            return (int)Math.round(probPred1(x));
        }

        /** This function takes a test set as input, call the predict() to predict a label for it, and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix **/
        public void printPerformance(List<LRInstance> testInstances) {
            double acc = 0;
            double p_pos = 0, r_pos = 0, f_pos = 0;
            double p_neg = 0, r_neg = 0, f_neg = 0;
            int TP=0, TN=0, FP=0, FN=0; // TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives

            // TODO: write code here to compute the above mentioned variables

            for(LRInstance instance : testInstances){
                double prediction = predict(instance.x);

                // prediction is accurate
                if(prediction == instance.label){
                    if(prediction == 1) {
                        TP++;
                    }else{
                        TN++;
                    }
                }else { // prediction inaccurate
                    if (prediction == 1) {
                        FP++;
                    } else {
                        FN++;
                    }
                }
            }

            acc = ((double)TP+TN)/testInstances.size();

            p_pos = TP/((double)TP+FP);
            r_pos = TP/((double)TP+FN);
            f_pos = 2*(p_pos+r_pos)/(p_pos+r_pos);

            p_neg = TN/((double)TN+FN);
            r_neg = TN/((double)TN+FP);
            f_neg = 2*(p_neg+r_neg)/(p_neg+r_neg);

            System.out.println("Accuracy="+acc);
            System.out.println("P, R, and F1 score of the positive class=" + p_pos + " " + r_pos + " " + f_pos);
            System.out.println("P, R, and F1 score of the negative class=" + p_neg + " " + r_neg + " " + f_neg);
            System.out.println("Confusion Matrix");
            System.out.println(TP + "\t" + FN);
            System.out.println(FP + "\t" + TN);
        }


        /** Train the Logistic Regression using Stochastic Gradient Ascent **/
        /** Also compute the log-likelihood of the data in this function **/
        public void train(List<LRInstance> instances) {
            for (int n = 0; n < ITERATIONS; n++) {
                double lik = 0.0; // Stores log-likelihood of the training data for this iteration
                for (int i=0; i < instances.size(); i++) {
                    // TODO: Train the model with bias

                    // our weights are already initialized as zero

                    double[] feats = instances.get(i).x;
                    int label = instances.get(i).label;
                    double prob = probPred1(feats);
                    double dotProduct = 0;

                    // for each of the weights (parameter w_i)
                    for(int p = 1; p < weights.length; p++) {
                        weights[p] = weights[p] + ( rate * feats[p-1] * (label - prob) );
                        weights[0] = weights[0] + ( rate * (label - prob));
                        dotProduct += weights[p] * feats[p-1];
                    }

                    // TODO: Compute the log-likelihood of the data here. Remember to take logs when necessary

                    lik += label * dotProduct - Math.log(1 + Math.exp(dotProduct));
                }
                System.out.println("iteration: " + n + " lik: " + lik);
            }
        }

        public static class LRInstance {
            public int label; // Label of the instance. Can be 0 or 1
            public double[] x; // The feature vector for the instance

            /** TODO: Constructor for initializing the Instance object **/
            public LRInstance(int label, double[] x) {
                this.label = label;
                this.x = x;
            }
        }

        /** Function to read the input dataset **/
        public static List<LRInstance> readDataSet(String file) throws FileNotFoundException {
            List<LRInstance> dataset = new ArrayList<LRInstance>();
            Scanner scanner = null;
            try {
                scanner = new Scanner(new File(file));

                while(scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    if (line.startsWith("ju")) { // Ignore the header line
                        continue;
                    }
                    String[] columns = line.replace("\n", "").split(",");

                    // every line in the input file represents an instance-label pair
                    int i = 0;
                    double[] data = new double[columns.length - 1];
                    for (i=0; i < columns.length - 1; i++) {
                        data[i] = Double.valueOf(columns[i]);
                    }
                    int label = Integer.parseInt(columns[i]); // last column is the label
                    LRInstance instance = new LRInstance(label, data); // create the instance
                    dataset.add(instance); // add instance to the corpus
                }
            } finally {
                if (scanner != null)
                    scanner.close();
            }
            return dataset;
        }


        public static void main(String... args) throws FileNotFoundException {
            List<LRInstance> trainInstances = readDataSet("HW4_trainset.csv");
            List<LRInstance> testInstances = readDataSet("HW4_testset.csv");

            // create an instance of the classifier
            int d = trainInstances.get(0).x.length;
            LogisticRegression_withBias logistic = new LogisticRegression_withBias(d);

            logistic.train(trainInstances);

            System.out.println("Norm of the learned weights = "+logistic.weightsL2Norm());
            System.out.println("Length of the weight vector = "+logistic.weights.length);

            // printing accuracy for different values of lambda
            System.out.println("-----------------Printing train set performance-----------------");
            logistic.printPerformance(trainInstances);

            System.out.println("-----------------Printing test set performance-----------------");
            logistic.printPerformance(testInstances);
        }

    }
