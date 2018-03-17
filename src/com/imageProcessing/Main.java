package com.imageProcessing;

import com.imageProcessing.network.DataGenerator;
import com.imageProcessing.network.Network;
import com.imageProcessing.network.NetworkTools;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.*;

public class Main {
    private static final int NUMBER_OF_INPUTS = 7921;
    private static double[][] inputs = new double[NUMBER_OF_INPUTS][];
    private static double[][] targets = new double[NUMBER_OF_INPUTS][];

    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}

    static final FilenameFilter IMAGE_FILTER = new FilenameFilter() {
        @Override
        public boolean accept(File dir, String name) {
            if(name.endsWith(".jpg")||name.endsWith(".jpeg"))
                return true;
            return false;
        }
    };

    public static void main(String[] args) throws IOException {
        Network network = Network.load("NetworkFile");
        doTheMagic(network, 10000, 0.3);
        NetworkTools.saveAsImage(network, 1);

    }

    private static void doTheMagic(Network network, int trainingIterations, double eta) throws IOException {
        for(int i=0; i<trainingIterations; i++){
            for(int j=0; j<inputs.length; j++)
                network.train(inputs[j], targets[j], eta);
            System.out.println((int)(((double)i+1)/((double)trainingIterations)*100)+"% finished");
        }
        network.save("NetworkFile");
    }


    private static void loadInputsAndOutputs() throws IOException {
        File inputsFile = new File("inputs.csv");
        FileReader fr = new FileReader(inputsFile);
        BufferedReader br = new BufferedReader(fr);
        for(int i = 0; i< NUMBER_OF_INPUTS; i++){
            inputs[i] = new double[1024];
            String line = br.readLine();
            String[] values = line.split(";");
            for(int j=0; j<1024; j++){
                inputs[i][j] = Double.valueOf(values[j]);
            }
        }
        br.close();
        fr.close();

        File targetsFile = new File("targets.csv");
        fr = new FileReader(targetsFile);
        br = new BufferedReader(fr);
        for(int i = 0; i< NUMBER_OF_INPUTS; i++){
            targets[i] = new double[2];
            String line = br.readLine();
            String[] values = line.split(";");
            for(int j=0; j<2; j++){
                targets[i][j] = Double.valueOf(values[j]);
            }
        }
        br.close();
        fr.close();
    }


    private static void createMatchups() throws IOException {
        System.out.println("Generating values...");
        File inputs = new File("inputs.csv");
        File targets = new File("targets.csv");
        FileWriter fw1 = new FileWriter(inputs);
        BufferedWriter inputsWriter = new BufferedWriter(fw1);
        FileWriter fw2 = new FileWriter(targets);
        BufferedWriter targetsWriter = new BufferedWriter(fw2);

        for(int i=1; i<=41; i++){
            File dir = new File("training pictures\\"+String.valueOf(i));
            File[] files = dir.listFiles();
            for(int j=0; j<files.length; j++){
                if(files[j].isFile()) {

                    Mat image1 = Processing.getImage(String.valueOf(j), String.valueOf(i));
                    Imgproc.cvtColor(image1, image1, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.GaussianBlur(image1, image1, new Size(5, 5), 0);
                    for(int folder=1; folder<=41; folder++){
                        File innerFolder = new File("training pictures\\"+String.valueOf(folder));
                        File[] innerFiles = innerFolder.listFiles();
                        for(int file=0; file<innerFiles.length; file++){
                             //&& !(inf==i && inFi==j) add this condition if don't need the same images
                            if(innerFiles[file].isFile()) {
                                Mat image2 = Processing.getImage(String.valueOf(file), String.valueOf(folder));
                                Imgproc.cvtColor(image2, image2, Imgproc.COLOR_BGR2GRAY);
                                Imgproc.GaussianBlur(image2, image2, new Size(5, 5), 0);

                                float[] values = DataGenerator.generate(image1, image2);

                                for(int v=0; v<values.length; v++){
                                    inputsWriter.write(String.valueOf(values[v]));
                                    if(v!=values.length-1)
                                        inputsWriter.write(";");
                                }
                                inputsWriter.newLine();

                                if(folder==i){
                                    targetsWriter.write("1;0");
                                    targetsWriter.newLine();
                                } else {
                                    targetsWriter.write("0;1");
                                    targetsWriter.newLine();
                                }
                                System.gc();

                            }
                        }
                    }
                }
            }
            System.out.println(i);
        }
        inputsWriter.close();
        fw1.close();
        targetsWriter.close();
        fw2.close();
    }
}
