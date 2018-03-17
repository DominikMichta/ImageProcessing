package com.imageProcessing.network;

import java.io.*;

public class Network{

    private double[][] output;
    double[][][] weights;
    private double[][] bias;

    private double[][] error_signal;
    private double[][] output_derivative;

    int[] NETWORK_LAYER_SIZES;
    int INPUT_SIZE;
    int OUTPUT_SIZE;
    int NETWORK_SIZE;

    public Network(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE-1];

        this.output = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];

        for(int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];

            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], -0.5,0.7);

            if(i > 0) {
                weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i],NETWORK_LAYER_SIZES[i-1], -1,1);
            }
        }
    }

    public double[] calculate(double... input) {
        if(input.length != this.INPUT_SIZE) return null;
        this.output[0] = input;
        for(int layer = 1; layer < NETWORK_SIZE; layer ++) {
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron ++) {

                double sum = bias[layer][neuron];
                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron ++) {
                    sum += output[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = sigmoid(sum);
                //Sigmoid Derivative
                output_derivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);

                //ReLu Derivative
                //output_derivative[layer][neuron] = sum>0 ? 1d : 0d;
            }
        }
        return output[NETWORK_SIZE-1];
    }

    public void train(double[] input, double[] target, double eta) {
        if(input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) return;
        calculate(input);
        backpropError(target);
        updateWeights(eta);
    }

    public void backpropError(double[] target) {
        for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE-1]; neuron ++) {
            error_signal[NETWORK_SIZE-1][neuron] = (output[NETWORK_SIZE-1][neuron] - target[neuron])
                    * output_derivative[NETWORK_SIZE-1][neuron];
        }
        for(int layer = NETWORK_SIZE-2; layer > 0; layer --) {
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron ++){
                double sum = 0;
                for(int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer+1]; nextNeuron ++) {
                    sum += weights[layer + 1][nextNeuron][neuron] * error_signal[layer + 1][nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * output_derivative[layer][neuron];
            }
        }
    }

    public void updateWeights(double eta) {
        for(int layer = 1; layer < NETWORK_SIZE; layer++) {
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {

                double delta = - eta * error_signal[layer][neuron];
                bias[layer][neuron] += delta;

                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer-1]; prevNeuron ++) {
                    weights[layer][neuron][prevNeuron] += delta * output[layer-1][prevNeuron];
                }
            }
        }
    }

    private double sigmoid( double x) {
        return 1d / ( 1 + Math.exp(-x));
    }
    private double reLU( double x) {
        return x<0 ? 0d : x;
    }


    public void save(String filename) throws IOException {
        File networkFile = new File("C:\\Users\\micht\\IdeaProjects\\OpenCv\\src\\pictures\\"+filename+".csv");
        FileWriter fw = new FileWriter(networkFile);
        BufferedWriter bw = new BufferedWriter(fw);


        bw.write(String.valueOf(NETWORK_LAYER_SIZES.length));
        bw.newLine();
        for(int i=0; i<NETWORK_LAYER_SIZES.length; i++)
            bw.write(NETWORK_LAYER_SIZES[i]+";");
        bw.newLine();
        bw.write(INPUT_SIZE+";"+OUTPUT_SIZE+";"+NETWORK_SIZE);
        bw.newLine();

        for(int i = 0; i < NETWORK_SIZE; i++) {
            int biasSize = NETWORK_LAYER_SIZES[i];
            for(int j=0; j<biasSize; j++){
                bw.write(String.valueOf(this.bias[i][j])+";");
            }
            bw.newLine();
        }
        for(int i = 0; i < NETWORK_SIZE; i++) {
            if(i > 0) {
                int sizeX = NETWORK_LAYER_SIZES[i];
                int sizeY = NETWORK_LAYER_SIZES[i-1];
                bw.write(sizeX+";"+sizeY+";");
                bw.newLine();
                for(int x=0; x<sizeX; x++){
                    for(int y=0; y<sizeY; y++){
                        bw.write(String.valueOf(this.weights[i][x][y])+";");
                    }
                    bw.newLine();
                }
            }
        }

        bw.close();
        fw.close();
    }


    public static Network load(String filename) throws IOException {
        File networkFile = new File("C:\\Users\\micht\\IdeaProjects\\OpenCv\\src\\pictures\\"+filename+".csv");
        FileReader fr = new FileReader(networkFile);
        BufferedReader br = new BufferedReader(fr);

        int networkLayerSizesLength = Integer.valueOf(br.readLine());
        String line = br.readLine();
        String[] networkLayerSizes = line.split(";");
        int[] NetworkLayerSizes = new int[networkLayerSizesLength];
        for(int i=0; i<networkLayerSizesLength; i++)
            NetworkLayerSizes[i]=Integer.valueOf(networkLayerSizes[i]);
        Network ret = new Network(NetworkLayerSizes);
        line = br.readLine();
        String[] sizes = line.split(";");
        int INPUT_SIZE = Integer.valueOf(sizes[0]);
        int OUTPUT_SIZE = Integer.valueOf(sizes[1]);
        int NETWORK_SIZE = Integer.valueOf(sizes[2]);


        for(int i = 0; i < NETWORK_SIZE; i++) {
            int biasSize = NetworkLayerSizes[i];
            line = br.readLine();
            String[] biases = line.split(";");
            for(int j=0; j<biasSize; j++){
                ret.bias[i][j] = Double.valueOf(biases[j]);
            }
        }
        for(int i = 0; i < NETWORK_SIZE; i++) {
            if(i > 0) {
                line = br.readLine();
                String[] weightsSizes = line.split(";");
                int sizeX = Integer.valueOf(weightsSizes[0]);
                int sizeY = Integer.valueOf(weightsSizes[1]);

                for(int x=0; x<sizeX; x++){
                    line = br.readLine();
                    String[] weights = line.split(";");
                    for(int y=0; y<sizeY; y++){
                        ret.weights[i][x][y] = Double.valueOf(weights[y]);
                    }
                }
            }
        }

        br.close();
        fr.close();
        return ret;
    }

}
