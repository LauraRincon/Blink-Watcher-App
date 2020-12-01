package com.blinkwatcher;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase mOpenCvCameraView;
    BaseLoaderCallback mLoaderCallback;

    public static final int        JAVA_DETECTOR       = 0;
    private static final String    TAG                 = "<BLINK-Watcher>";
    private int                    mDetectorType       = JAVA_DETECTOR;
    private File                   mCascadeFile;
    private File                   mCascadeFileEye;
    private CascadeClassifier      mJavaDetector;
    private CascadeClassifier      mJavaDetectorEye;
    Button StartDrivingBtn;

    boolean startDriving = false;

    Net detector;

    public void loadModel(View Button) {
        StartDrivingBtn = (Button) findViewById(R.id.btn_StartDiving);
        if (!startDriving) {
            startDriving = true;
            String protoPath = getPath("deploy.prototxt", this);
            String caffeWeights = getPath("res10_300x300_ssd_iter_140000.caffemodel", this);
            detector = Dnn.readNetFromCaffe(protoPath, caffeWeights);
            StartDrivingBtn.setText("Stop driving");
        }
        else{
            startDriving = false;
            StartDrivingBtn.setText("Start driving");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mLoaderCallback = new BaseLoaderCallback(this){
            @Override
            public void onManagerConnected(int status){
                super.onManagerConnected(status);
                switch(status){
                    case BaseLoaderCallback.SUCCESS:
                        try {
                            // load cascade file from application resources
                            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                            mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                            FileOutputStream os = new FileOutputStream(mCascadeFile);

                            byte[] buffer = new byte[4096];
                            int bytesRead;
                            while ((bytesRead = is.read(buffer)) != -1) {
                                os.write(buffer, 0, bytesRead);
                            }
                            is.close();
                            os.close();

                            // load cascade file from application resources
                            InputStream ise = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                            File cascadeDirEye = getDir("cascade", Context.MODE_PRIVATE);
                            mCascadeFileEye = new File(cascadeDirEye, "haarcascade_lefteye_2splits.xml");
                            FileOutputStream ose = new FileOutputStream(mCascadeFileEye);

                            while ((bytesRead = ise.read(buffer)) != -1) {
                                ose.write(buffer, 0, bytesRead);
                            }
                            ise.close();
                            ose.close();

                            mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                            if (mJavaDetector.empty()) {
                                Log.e(TAG, "Failed to load cascade classifier");
                                mJavaDetector = null;
                            } else
                                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                            mJavaDetectorEye = new CascadeClassifier(mCascadeFileEye.getAbsolutePath());
                            if (mJavaDetectorEye.empty()) {
                                Log.e(TAG, "Failed to load cascade classifier for eye");
                                mJavaDetectorEye = null;
                            } else
                                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFileEye.getAbsolutePath());

                            cascadeDir.delete();
                            cascadeDirEye.delete();

                        } catch (IOException e) {
                            e.printStackTrace();
                            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                        }
                        mOpenCvCameraView.enableFpsMeter();
                        mOpenCvCameraView.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat mRgba = inputFrame.rgba();

        if (startDriving) {
            Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2RGB);
            Mat imageBlog = Dnn.blobFromImage(mRgba, 1.0, new Size(300, 300), new Scalar(104.0, 177.0, 123.0), true, false, CvType.CV_32F);

            detector.setInput(imageBlog);
            Mat detections = detector.forward();

            int cols = mRgba.cols();
            int rows = mRgba.rows();
            double THRESHOLD = 0.80;

            detections = detections.reshape(1, (int)detections.total() / 7);
            for (int i = 0; i < detections.rows(); ++i) {

                double confidence = detections.get(i, 2)[0];
                if (confidence > THRESHOLD) {
                    int left   = (int)(detections.get(i, 3)[0] * cols);
                    int top    = (int)(detections.get(i, 4)[0] * rows);
                    int right  = (int)(detections.get(i, 5)[0] * cols);
                    int bottom = (int)(detections.get(i, 6)[0] * rows);
                    Imgproc.rectangle(mRgba, new Point(left, top), new Point(right, bottom), new Scalar(255, 255, 0), 2);
                    //Start detecting the eyes
                }
            }
        }

        return mRgba;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        }
        else
        {
            mLoaderCallback.onManagerConnected(mLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(mOpenCvCameraView!=null){
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(mOpenCvCameraView!=null){
            mOpenCvCameraView.disableView();
        }
    }

    // Upload file to storage and return a path.
    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }
}