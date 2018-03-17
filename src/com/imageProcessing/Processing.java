package com.imageProcessing;

import com.imageProcessing.network.DataGenerator;
import com.imageProcessing.network.Network;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.*;
import java.util.concurrent.CountDownLatch;

public class Processing {

    public static Mat loadImage(String filename, String folder){
        //
        // File input = new File("C:\\Users\\micht\\IdeaProjects\\OpenCv\\src\\pictures\\"+folder+"\\" + filename+".jpg");
        File input = new File("C:\\Users\\micht\\Desktop\\training data\\"+folder+"\\" + filename+".jpg");
        //File input = new File("C:\\Users\\micht\\Desktop\\training data" + filename+".jpg");
        BufferedImage image = null;
        try {
            image = ImageIO.read(input);
        } catch (IOException e) {
            input = new File("C:\\Users\\micht\\IdeaProjects\\OpenCv\\src\\pictures\\"+folder+"\\" + filename+".jpeg");
            try {
                image = ImageIO.read(input);
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        }
        BufferedImage imageCopy =
                new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        imageCopy.getGraphics().drawImage(image, 0, 0, null);
        byte[] data = ((DataBufferByte) imageCopy.getRaster().getDataBuffer()).getData();
        Mat img = new Mat(image.getHeight(),image.getWidth(), CvType.CV_8UC3);
        img.put(0, 0, data);
        return img;
    }

    public static Mat getImage(String filename1, String folder1){
        return loadImage(filename1, folder1);
    }

    public static double[] compare(String filename1, String filename2, Network network){
        Mat image1 = loadImage(filename1, "temp2");
        Mat image2 = loadImage(filename2, "temp2");
        Imgproc.cvtColor(image1, image1, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(image1, image1, new Size(5, 5), 0);
        Imgproc.cvtColor(image2, image2, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(image2, image2, new Size(5, 5), 0);
        float[] values = DataGenerator.generate(image1, image2);
        double[] input = new double[values.length];
        for(int i=0; i<values.length; i++){
            input[i]=(double)values[i];
        }
        return network.calculate(input);
    }

    public static Mat[] partialImages(Mat image, int quantity){
        Mat[] smallImages = new Mat[quantity*quantity];
        for(int i=0; i<quantity; i++){
            for(int j=0; j<quantity; j++){
                smallImages[i*quantity+j]=new Mat(image, new Range((image.rows()/quantity)*i, (image.rows()/quantity)*(i+1)),
                        new Range((image.cols()/quantity)*j, (image.cols()/quantity)*(j+1)));
            }
        }
        return smallImages;
    }


    static void process(String fileName) throws IOException, InterruptedException {
        Mat image = loadImage(fileName, "temp");
        Mat originalImage = image.clone();
        Imgproc.pyrDown(image, image);
        Imgproc.pyrDown(image, image);
        Mat lab = new Mat();
        Imgproc.cvtColor(image, lab, Imgproc.COLOR_BGR2Lab);
        List<Mat> channels = new ArrayList<>();
        List<Mat> dstChannels = new ArrayList<Mat>(){{
            add(new Mat());
            add(new Mat());
            add(new Mat());
        }};
        Core.split(image, channels);

        CountDownLatch countDownLatch = new CountDownLatch(3);
        Thread[] threads = new Thread[3];
        for(int i=0; i<3; i++){
            threads[i] = new Thread(new BilateralFilterRunnable(channels.get(i), dstChannels.get(i), countDownLatch));
            threads[i].start();
        }
        countDownLatch.await();

        Core.subtract(dstChannels.get(0), dstChannels.get(2), dstChannels.get(0));

        Imgproc.threshold(dstChannels.get(0), dstChannels.get(0), 0, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.dilate(dstChannels.get(0), dstChannels.get(0), kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(dstChannels.get(0), contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        MatOfInt hull = new MatOfInt();
        for(int i=0; i<contours.size(); i++){
            Imgproc.convexHull(contours.get(i), hull, false);
            contours.set(i, convertIndexesToPoints(contours.get(i), hull));
        }

        MatOfPoint biggest = new MatOfPoint();
        double max_area = 0;
        MatOfPoint2f tempMat=new MatOfPoint2f();
        for(MatOfPoint contour: contours){
            double area = Imgproc.contourArea(contour);
            if(area>1000 && area>max_area){
                double peri = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
                Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), tempMat, 0.04*peri, true);
                if(new MatOfPoint(tempMat.toArray()).toArray().length==4){
                    biggest = new MatOfPoint(tempMat.toArray());
                    max_area = area;
                }
            }
        }

        List<MatOfPoint> contour = new ArrayList<>();
        contour.add(biggest);
        MatOfPoint2f maxMatOfPoint2f = new MatOfPoint2f(biggest.toArray());
        RotatedRect rect;
        try {
            rect = Imgproc.minAreaRect(maxMatOfPoint2f);
        }catch (Exception e){
            return;
        }

        /*
        for (int i = 0; i < 4; ++i)
        {
            Core.line(background, points[i], points[(i + 1) % 4], new Scalar(255, 0, 0, 1), 4);
        }
        */
        //Highgui.imwrite("C:\\Users\\micht\\IdeaProjects\\OpenCv\\src\\pictures\\processingOutput\\" + fileName +".jpeg", background);

        Mat outputMat = perspectiveTransform(originalImage, rect);


        saveImage(outputMat, fileName, "output");
    }

    private static Mat perspectiveTransform(Mat originalImage, RotatedRect rect) {
        Point points[] = new Point[4];
        rect.points(points);
        int id = findStartId(points);

        for(int i=0; i<4; i++){
            points[i].x = points[i].x*4;
            points[i].y = points[i].y*4;
        }


        int resultWidth = (int)rect.size.width*4;
        int resultHeight = (int)rect.size.height*4;

        if(resultWidth>resultHeight){
            if(getDistance(points[id], points[(id+1)%4]) < getDistance(points[id], points[(id+3)%4])){
                id=(id+3)%4;
            }
        } else {
            if(getDistance(points[id], points[(id+1)%4]) > getDistance(points[id], points[(id+3)%4])){
                id=(id+3)%4;
            }
        }

        Mat outputMat = new Mat(resultWidth, resultHeight, CvType.CV_8UC3);

        Point ocvPIn1 = new Point(points[id].x, points[id].y);
        Point ocvPIn2 = new Point(points[(id+1)%4].x, points[(id+1)%4].y);
        Point ocvPIn3 = new Point(points[(id+2)%4].x, points[(id+2)%4].y);
        Point ocvPIn4 = new Point(points[(id+3)%4].x, points[(id+3)%4].y);
        List<Point> source = new ArrayList<>();
        source.add(ocvPIn1);
        source.add(ocvPIn2);
        source.add(ocvPIn3);
        source.add(ocvPIn4);
        Mat startM = Converters.vector_Point2f_to_Mat(source);
        Point ocvPOut1, ocvPOut2, ocvPOut3, ocvPOut4;
        ocvPOut1 = new Point(0, 0);
        ocvPOut2 = new Point(resultWidth, 0);
        ocvPOut3 = new Point(resultWidth, resultHeight);
        ocvPOut4 = new Point(0, resultHeight);
        List<Point> dest = new ArrayList<>();
        dest.add(ocvPOut1);
        dest.add(ocvPOut2);
        dest.add(ocvPOut3);
        dest.add(ocvPOut4);
        Mat endM = Converters.vector_Point2f_to_Mat(dest);

        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(startM, endM);

        Imgproc.warpPerspective(originalImage,
                outputMat,
                perspectiveTransform,
                new Size(resultWidth, resultHeight),
                Imgproc.INTER_CUBIC);
        return outputMat;
    }

    private static MatOfPoint convertIndexesToPoints(MatOfPoint contour, MatOfInt indexes) {
        int[] arrIndex = indexes.toArray();
        Point[] arrContour = contour.toArray();
        Point[] arrPoints = new Point[arrIndex.length];

        for (int i=0;i<arrIndex.length;i++) {
            arrPoints[i] = arrContour[arrIndex[i]];
        }

        MatOfPoint hull = new MatOfPoint();
        hull.fromArray(arrPoints);
        return hull;
    }


    public static void saveImage(Mat image, String filename, String folder){
        if(folder!=null)
            Imgcodecs.imwrite("C:\\Users\\micht\\IdeaProjects\\OpenCv\\src\\pictures\\"+folder+"\\"+
                    filename+".jpeg", image);
        else
            Imgcodecs.imwrite("C:\\Users\\micht\\IdeaProjects\\OpenCv\\src\\pictures\\"+filename +".jpeg", image);
    }

    private static int findStartId(Point[] points){
        int id = 0;
        double minDistance = Double.MAX_VALUE;
        Point start = new Point(0, 0);
        for(int i=0; i<4; i++){
            double distance = getDistance(start, points[i]);
            if(distance<minDistance){
                minDistance = distance;
                id = i;
            }
        }
        return id;
    }

    private static double getDistance(Point A, Point B){
        return Math.sqrt(Math.pow((A.x-B.x), 2)+Math.pow((A.y-B.y), 2));
    }

}
