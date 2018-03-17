package com.imageProcessing.network;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class NetworkTools {

    public static double[] createArray(int size, double init_value){
        if(size < 1){
            return null;
        }
        double[] ar = new double[size];
        for(int i = 0; i < size; i++){
            ar[i] = init_value;
        }
        return ar;
    }

    public static double[] createRandomArray(int size, double lower_bound, double upper_bound){
        if(size < 1){
            return null;
        }
        double[] ar = new double[size];
        for(int i = 0; i < size; i++){
            ar[i] = randomValue(lower_bound,upper_bound);
        }
        return ar;
    }

    public static double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound){
        if(sizeX < 1 || sizeY < 1){
            return null;
        }
        double[][] ar = new double[sizeX][sizeY];
        for(int i = 0; i < sizeX; i++){
            ar[i] = createRandomArray(sizeY, lower_bound, upper_bound);
        }
        return ar;
    }

    public static double randomValue(double lower_bound, double upper_bound){
        return Math.random()*(upper_bound-lower_bound) + lower_bound;
    }

    public static void saveAsImage(Network network, int parts){
        final int radius = 50;
        Mat image = new Mat(new Size(12000, 12000), CvType.CV_8UC3);
        image.setTo(new Scalar(185, 128, 41));
        int[][] coordinates = new int[network.NETWORK_SIZE][];
        for(int layer=0; layer<network.NETWORK_SIZE; layer++){
            int neuronsCount = network.NETWORK_LAYER_SIZES[layer];
            if(layer==0)
                neuronsCount/=parts;

            int d=image.rows()/(neuronsCount+1);
            coordinates[layer] = new int[neuronsCount];
            for(int neuron=0; neuron<neuronsCount; ++neuron){
                coordinates[layer][neuron]=d;
                d+=image.rows()/(neuronsCount+1);
            }
        }

        for(int i=0; i<coordinates.length; i++){
            int x = (image.cols()/coordinates.length)*i+(image.cols()/(coordinates.length*2));
            for(int j=0; j<coordinates[i].length; j++){
                Imgproc.circle(image, new Point(x, coordinates[i][j]), radius, new Scalar(0, 0, 255), -1);
            }
        }

        for(int i=network.NETWORK_SIZE-1; i>0; --i){
            int x1 = (image.cols()/coordinates.length)*i+(image.cols()/(coordinates.length*2));
            int x2 = x1-(image.cols()/coordinates.length);
            for(int neuron=0; neuron<network.NETWORK_LAYER_SIZES[i]; neuron++){
                int prevNeuronsCount = network.NETWORK_LAYER_SIZES[i-1];
                if(i==1)
                    prevNeuronsCount/=parts;
                for(int prevNeuron=0; prevNeuron<prevNeuronsCount; prevNeuron++){
                    int shade = (int)((network.weights[i][neuron][prevNeuron] / Math.sqrt( 1 + Math.pow(network.weights[i][neuron][prevNeuron], 2)))*128);
                    Scalar color = new Scalar(128 + shade, 128 + shade, 128 + shade);
                    Imgproc.line(image, new Point(x1, coordinates[i][neuron]), new Point(x2, coordinates[i - 1][prevNeuron]),
                            color, 2);
                }
            }
        }

        Imgcodecs.imwrite("network.bmp", image);

    }
}
