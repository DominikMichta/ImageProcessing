package com.imageProcessing.network;

import com.imageProcessing.Processing;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class DataGenerator {
    static Mat tmp1 = new Mat();
    static Mat tmp2 = new Mat();
    static Mat[] smallImages = null;
    public static float[] generate(Mat image1, Mat image2){
        if(image1.rows()>image2.rows()){
            Imgproc.resize(image1, tmp1, new Size(image1.rows(), image1.rows()));
            Imgproc.resize(image2, tmp2, new Size(image1.rows(), image1.rows()));
        } else {
            Imgproc.resize(image1, tmp1, new Size(image2.rows(), image2.rows()));
            Imgproc.resize(image2, tmp2, new Size(image2.rows(), image2.rows()));
        }

        Core.absdiff(tmp1, tmp2, tmp1);
        int q = tmp1.rows()/1024;
        if(q!=0){
            Imgproc.resize(tmp1, tmp1, new Size(1024*q, 1024*q));
        } else {
            Imgproc.resize(tmp1, tmp1, new Size(1024, 1024));
        }

        Mat smallImage = findLargestDifference(tmp1);

        return getData(smallImage);
    }

    private static Mat findLargestDifference(Mat image){
        double biggestAverage = Double.MIN_VALUE;
        int imageIndex=0;
        smallImages = Processing.partialImages(image, 8);
        for(int i=0; i<64; i++){
            Scalar s = Core.mean(smallImages[i]);
            if((s.val[0]+s.val[1]+s.val[2])>biggestAverage){
                biggestAverage = s.val[0]+s.val[1]+s.val[2];
                imageIndex=i;
            }
        }
        return smallImages[imageIndex];
    }

    private static float[] getData(Mat image){

        Mat histogram = new Mat();
        List<Mat> channels = new ArrayList<>();
        Core.split(image, channels);
        float[] data = new float[256];
        Imgproc.calcHist(channels, new MatOfInt(0), new Mat(), histogram,
                new MatOfInt(256), new MatOfFloat(0f, 256f), false);
        Core.normalize(histogram, histogram, 0, 1, Core.NORM_MINMAX);
        histogram.get(0, 0, data);
        return data;
    }
}
