package com.imageProcessing;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.concurrent.CountDownLatch;

public class BilateralFilterRunnable implements Runnable {
    Mat channel;
    Mat dstChannel;
    CountDownLatch done;

    public BilateralFilterRunnable(Mat channel, Mat dstChannel, CountDownLatch done){
        this.channel = channel;
        this.dstChannel = dstChannel;
        this.done = done;

    }

    @Override
    public void run() {
        int t=10;
        Imgproc.bilateralFilter(channel, dstChannel, t, t*2, t/2);
        done.countDown();
    }
}
