    package com.example.ocr1;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.ContentValues;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.util.SparseArray;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.theartofdev.edmodo.cropper.CropImage;
import com.theartofdev.edmodo.cropper.CropImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

    public class MainActivity extends AppCompatActivity {

        private static final int CAMERA_REQUEST_CODE=200;
        private static final int IMAGE_PICK_CAMERA_CODE=1001;
        private static final int STORAGE_REQUEST_CODE=400;
        private static final int IMAGE_PICK_GALLERY_CODE=1000;
        private static Bitmap bmp, yourSelectedImage, bmpimg1, bmpimg2;
        private static String descriptorType;
        private static long startTime, endTime;
        private static InputStream imageStream;
        private static TextView percentDetect;
        private static int descriptor = DescriptorExtractor.BRISK;
        private static Uri selectedImage;
        String storagePermission[];
        String cameraPermission[];
        Uri immage_uri;
        ImageView mPreviewIv;
        ImageView mPreviewIv2;
        public static Button start;
        public boolean btnB=false;
        private static int min_dist = 10;
        private static int min_matches = 750;
        private static String text;
        private static String path1, path2;
        private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS: {
                        Log.i("Main Activity", "OpenCV loaded successfully");
                    }
                    break;
                    default: {
                        super.onManagerConnected(status);
                    }
                    break;
                }
            }
        };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mPreviewIv=findViewById(R.id.imageIv);
        mPreviewIv2=findViewById(R.id.imageIv2);
        start=findViewById(R.id.start);
        percentDetect=findViewById(R.id.percentDetect);
        cameraPermission=new String[]{Manifest.permission.CAMERA,
                Manifest.permission.WRITE_EXTERNAL_STORAGE};
        storagePermission=new String[]{
                Manifest.permission.WRITE_EXTERNAL_STORAGE};

       findViewById(R.id.Camera).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!checkCameraPermission()){
                    requestCameraPermission();

                }else {
                    pickCamera();
                    btnB=true;
                }

            }
        });
       findViewById(R.id.Camera2).setOnClickListener(new View.OnClickListener() {
           @Override
           public void onClick(View v) {
               if(!checkStoragePermission()){
                   requestStoragePermission();
               }else{
                   pickGallery();
                   btnB=false;
               }
           }
       });

       start.setOnClickListener(new View.OnClickListener() {
           @Override
           public void onClick(View v) {

               if (bmpimg1 != null && bmpimg2 != null) {
					/*if(bmpimg1.getWidth()!=bmpimg2.getWidth()){
						bmpimg2 = Bitmap.createScaledBitmap(bmpimg2, bmpimg1.getWidth(), bmpimg1.getHeight(), true);
					}*/
                   bmpimg1 = Bitmap.createScaledBitmap(bmpimg1, 100, 100, true);
                   bmpimg2 = Bitmap.createScaledBitmap(bmpimg2, 100, 100, true);
                   Mat img1 = new Mat();
                   Utils.bitmapToMat(bmpimg1, img1);
                   Mat img2 = new Mat();
                   Utils.bitmapToMat(bmpimg2, img2);
                   Imgproc.cvtColor(img1, img1, Imgproc.COLOR_RGBA2GRAY);
                   Imgproc.cvtColor(img2, img2, Imgproc.COLOR_RGBA2GRAY);
                   img1.convertTo(img1, CvType.CV_32F);
                   img2.convertTo(img2, CvType.CV_32F);
                   //Log.d("ImageComparator", "img1:"+img1.rows()+"x"+img1.cols()+" img2:"+img2.rows()+"x"+img2.cols());
                   Mat hist1 = new Mat();
                   Mat hist2 = new Mat();
                   MatOfInt histSize = new MatOfInt(180);
                   MatOfInt channels = new MatOfInt(0);
                   ArrayList<Mat> bgr_planes1= new ArrayList<Mat>();
                   ArrayList<Mat> bgr_planes2= new ArrayList<Mat>();
                   Core.split(img1, bgr_planes1);
                   Core.split(img2, bgr_planes2);
                   MatOfFloat histRanges = new MatOfFloat (0f, 180f);
                   boolean accumulate = false;
                   Imgproc.calcHist(bgr_planes1, channels, new Mat(), hist1, histSize, histRanges, accumulate);
                   Core.normalize(hist1, hist1, 0, hist1.rows(), Core.NORM_MINMAX, -1, new Mat());
                   Imgproc.calcHist(bgr_planes2, channels, new Mat(), hist2, histSize, histRanges, accumulate);
                   Core.normalize(hist2, hist2, 0, hist2.rows(), Core.NORM_MINMAX, -1, new Mat());
                   img1.convertTo(img1, CvType.CV_32F);
                   img2.convertTo(img2, CvType.CV_32F);
                   hist1.convertTo(hist1, CvType.CV_32F);
                   hist2.convertTo(hist2, CvType.CV_32F);

                   double compare = Imgproc.compareHist(hist1, hist2, Imgproc.CV_COMP_CHISQR);
                   Log.d("ImageComparator", "compare: "+compare);
                   if(compare>0 && compare<1500) {


                       new asyncTask(MainActivity.this).execute();
                       double percent=(compare*100)/1500;
                       Toast.makeText(MainActivity.this, "Images may be possible duplicates, verifying", Toast.LENGTH_LONG).show();
                       int p=100-(int)percent;
                       percentDetect.setText(""+p+"%");
                   }
                   else if(compare==0){

                       percentDetect.setText("100%");
                       Toast.makeText(MainActivity.this, "Images are exact duplicates", Toast.LENGTH_LONG).show();}
                   else{
                       percentDetect.setText("0%");
                       Toast.makeText(MainActivity.this, "Images are not duplicates", Toast.LENGTH_LONG).show();

                   }

                   startTime = System.currentTimeMillis();
               } else
                   Toast.makeText(MainActivity.this,
                           "You haven't selected images.", Toast.LENGTH_LONG)
                           .show();
           }


       });
    }
        @Override
        public void onResume() {
            super.onResume();
            if(!OpenCVLoader.initDebug()){
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_4, this,
                        mLoaderCallback);
            }else {
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            }

        }

        public static class asyncTask extends AsyncTask<Void, Void, Void>{
            private static Mat img1, img2, descriptors, dupDescriptors;
            private static FeatureDetector detector;
            private static DescriptorExtractor DescExtractor;
            private static DescriptorMatcher matcher;
            private static MatOfKeyPoint keypoints, dupKeypoints;
            private static MatOfDMatch matches, matches_final_mat;
            private static ProgressDialog pd;
            private static boolean isDuplicate = false;
            private MainActivity asyncTaskContext=null;
            private static Scalar RED = new Scalar(255,0,0);
            private static Scalar GREEN = new Scalar(0,255,0);

            public asyncTask(MainActivity context_){
                asyncTaskContext=context_;
            }
            @Override
            protected void onPreExecute() {
                pd = new ProgressDialog(asyncTaskContext);
                pd.setIndeterminate(true);
                pd.setCancelable(true);
                pd.setCanceledOnTouchOutside(false);
                pd.setMessage("Processing...");
                pd.show();
            }

            @Override
            protected Void doInBackground(Void... voids) {
                compare();
                return null;
            }

            @Override
            protected void onPostExecute(Void aVoid) {
                try {
                    Mat img3 = new Mat();
                    MatOfByte drawnMatches = new MatOfByte();
                    Features2d.drawMatches(img1, keypoints, img2, dupKeypoints,
                            matches_final_mat, img3, GREEN, RED,  drawnMatches, Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);
                    bmp = Bitmap.createBitmap(img3.cols(), img3.rows(),
                            Bitmap.Config.ARGB_8888);
                    Imgproc.cvtColor(img3, img3, Imgproc.COLOR_BGR2RGB);
                    Utils.matToBitmap(img3, bmp);
                    List<DMatch> finalMatchesList = matches_final_mat.toList();
                    final int matchesFound=finalMatchesList.size();
                    endTime = System.currentTimeMillis();
                    if (finalMatchesList.size() > min_matches)// dev discretion for
                    // number of matches to
                    // be found for an image
                    // to be judged as
                    // duplicate
                    {
                        text = finalMatchesList.size()
                                + " matches were found. Possible duplicate image.\nTime taken="
                                + (endTime - startTime) + "ms";
                        isDuplicate = true;
                    } else {
                        text = finalMatchesList.size()
                                + " matches were found. Images aren't similar.\nTime taken="
                                + (endTime - startTime) + "ms";
                        isDuplicate = false;
                    }
                    pd.dismiss();
                    final android.app.AlertDialog.Builder alertDialog = new AlertDialog.Builder(
                            asyncTaskContext);
                    alertDialog.setTitle("Result");
                    alertDialog.setCancelable(false);
                    LayoutInflater factory = LayoutInflater.from(asyncTaskContext);
                    final View view = factory.inflate(R.layout.image_view, null);
                    ImageView matchedImages = (ImageView) view
                            .findViewById(R.id.finalImage);
                    matchedImages.setImageBitmap(bmp);
                    matchedImages.invalidate();
                    final CheckBox shouldBeDuplicate = (CheckBox) view
                            .findViewById(R.id.checkBox);
                    TextView message = (TextView) view.findViewById(R.id.message);
                    message.setText(text);
                    alertDialog.setView(view);
                    shouldBeDuplicate
                            .setText("These images are actually duplicates.");
                    alertDialog.setPositiveButton("Add to logs",
                            new DialogInterface.OnClickListener() {
                                public void onClick(DialogInterface dialog,
                                                    int which) {
                                    File logs = new File(Environment
                                            .getExternalStorageDirectory()
                                            .getAbsolutePath()
                                            + "/imageComparator/Data Logs.txt");
                                    FileWriter fw;
                                    BufferedWriter bw;
                                    try {
                                        fw = new FileWriter(logs, true);
                                        bw = new BufferedWriter(fw);
                                        bw.write("Algorithm used: "
                                                + descriptorType
                                                + "\nHamming distance: "
                                                + min_dist + "\nMinimum good matches: "+min_matches
                                                +"\nMatches found: "+matchesFound+"\nTime elapsed: "+(endTime-startTime)+"seconds\n"+ path1
                                                + " was compared to " + path2
                                                + "\n" + "Is actual duplicate: "
                                                + shouldBeDuplicate.isChecked()
                                                + "\nRecognized as duplicate: "
                                                + isDuplicate + "\n");
                                        bw.close();
                                        Toast.makeText(
                                                asyncTaskContext,
                                                "Logs updated.\nLog location: "
                                                        + Environment
                                                        .getExternalStorageDirectory()
                                                        .getAbsolutePath()
                                                        + "/imageComparator/Data Logs.txt",
                                                Toast.LENGTH_LONG).show();
                                    } catch (IOException e) {
                                        // TODO Auto-generated catch block
                                        // e.printStackTrace();
                                        try {
                                            File dir = new File(Environment
                                                    .getExternalStorageDirectory()
                                                    .getAbsolutePath()
                                                    + "/imageComparator/");
                                            dir.mkdirs();
                                            logs.createNewFile();
                                            logs = new File(
                                                    Environment
                                                            .getExternalStorageDirectory()
                                                            .getAbsolutePath()
                                                            + "/imageComparator/Data Logs.txt");
                                            fw = new FileWriter(logs, true);
                                            bw = new BufferedWriter(fw);
                                            bw.write("Algorithm used: "
                                                    + descriptorType
                                                    + "\nMinimum distance between keypoints: "
                                                    + min_dist + "\n" + path1
                                                    + " was compared to " + path2
                                                    + "\n"
                                                    + "Is actual duplicate: "
                                                    + shouldBeDuplicate.isChecked()
                                                    + "\nRecognized as duplicate: "
                                                    + isDuplicate + "\n");
                                            bw.close();
                                            Toast.makeText(
                                                    asyncTaskContext,
                                                    "Logs updated.\nLog location: "
                                                            + Environment
                                                            .getExternalStorageDirectory()
                                                            .getAbsolutePath()
                                                            + "/imageComparator/Data Logs.txt",
                                                    Toast.LENGTH_LONG).show();
                                        } catch (IOException e1) {
                                            // TODO Auto-generated catch block
                                            e1.printStackTrace();
                                        }

                                    }
                                }
                            });
                    alertDialog.show();
                } catch (Exception e) {
                    e.printStackTrace();
                    Toast.makeText(asyncTaskContext, e.toString(),
                            Toast.LENGTH_LONG).show();
                }
            }

            void compare() {
                try {
                    bmpimg1 = bmpimg1.copy(Bitmap.Config.ARGB_8888, true);
                    bmpimg2 = bmpimg2.copy(Bitmap.Config.ARGB_8888, true);
                    img1 = new Mat();
                    img2 = new Mat();
                    Utils.bitmapToMat(bmpimg1, img1);
                    Utils.bitmapToMat(bmpimg2, img2);
                    Imgproc.cvtColor(img1, img1, Imgproc.COLOR_BGR2RGB);
                    Imgproc.cvtColor(img2, img2, Imgproc.COLOR_BGR2RGB);
                    detector = FeatureDetector.create(FeatureDetector.PYRAMID_FAST);
                    DescExtractor = DescriptorExtractor.create(descriptor);
                    matcher = DescriptorMatcher
                            .create(DescriptorMatcher.BRUTEFORCE_HAMMING);

                    keypoints = new MatOfKeyPoint();
                    dupKeypoints = new MatOfKeyPoint();
                    descriptors = new Mat();
                    dupDescriptors = new Mat();
                    matches = new MatOfDMatch();
                    detector.detect(img1, keypoints);
                    Log.d("LOG!", "number of query Keypoints= " + keypoints.size());
                    detector.detect(img2, dupKeypoints);
                    Log.d("LOG!", "number of dup Keypoints= " + dupKeypoints.size());
                    // Descript keypoints
                    DescExtractor.compute(img1, keypoints, descriptors);
                    DescExtractor.compute(img2, dupKeypoints, dupDescriptors);
                    Log.d("LOG!", "number of descriptors= " + descriptors.size());
                    Log.d("LOG!",
                            "number of dupDescriptors= " + dupDescriptors.size());
                    // matching descriptors
                    matcher.match(descriptors, dupDescriptors, matches);
                    Log.d("LOG!", "Matches Size " + matches.size());
                    // New method of finding best matches
                    List<DMatch> matchesList = matches.toList();
                    List<DMatch> matches_final = new ArrayList<DMatch>();
                    for (int i = 0; i < matchesList.size(); i++) {
                        if (matchesList.get(i).distance <= min_dist) {
                            matches_final.add(matches.toList().get(i));
                        }
                    }

                    matches_final_mat = new MatOfDMatch();
                    matches_final_mat.fromList(matches_final);
                } catch (Exception e) {
                    e.printStackTrace();
                }

            }


        }

        private void pickCamera() {
            ContentValues values=new ContentValues();
            values.put(MediaStore.Images.Media.TITLE,"NewPic");
            values.put(MediaStore.Images.Media.DESCRIPTION,"Image To text");
            immage_uri=getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,values);
            Intent cameraIntent=new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT,immage_uri);
            startActivityForResult(cameraIntent,IMAGE_PICK_CAMERA_CODE);

        }

        private void requestCameraPermission() {
            ActivityCompat.requestPermissions(this,cameraPermission,CAMERA_REQUEST_CODE);
        }

        private boolean checkCameraPermission() {
            boolean result= ContextCompat.checkSelfPermission(this,
                    Manifest.permission.CAMERA)==(PackageManager.PERMISSION_GRANTED);
            boolean result1= ContextCompat.checkSelfPermission(this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)==(PackageManager.PERMISSION_GRANTED);
            return result && result1;
        }

        private void requestStoragePermission() {
            ActivityCompat.requestPermissions(this,storagePermission,STORAGE_REQUEST_CODE);
        }

        private boolean checkStoragePermission() {
            boolean result= ContextCompat.checkSelfPermission(this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)==(PackageManager.PERMISSION_GRANTED);
            return result;
        }
        private void pickGallery() {
            Intent intent=new Intent(Intent.ACTION_PICK);
            intent.setType("image/*");
            startActivityForResult(intent,IMAGE_PICK_GALLERY_CODE);
        }

        @Override
        public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            switch (requestCode){
                case CAMERA_REQUEST_CODE:
                    if (grantResults.length>0){
                        boolean cameraAccepted=grantResults[0]==PackageManager.PERMISSION_GRANTED;
                        boolean writeStorageAccepted=grantResults[0]==PackageManager.PERMISSION_GRANTED;
                        if (cameraAccepted && writeStorageAccepted){
                            pickCamera();
                        }else {
                            Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
                        }

                    }
                    break;
                case STORAGE_REQUEST_CODE:
                    if(grantResults.length>0){
                        boolean writeStorageAccepted = grantResults[0] ==
                                PackageManager.PERMISSION_GRANTED;
                        if (writeStorageAccepted ){
                            pickGallery();
                        }else {
                            Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
                        }

                    }
                    break;
            }
        }

        @Override
        protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
            super.onActivityResult(requestCode, resultCode, data);

            if (requestCode!=RESULT_OK){

                if (requestCode==IMAGE_PICK_GALLERY_CODE){

                    CropImage.activity(data.getData())
                            .setGuidelines(CropImageView.Guidelines.ON)
                            .start(this);
                    Log.d("gal",""+data.getData());

                }
                if (requestCode==IMAGE_PICK_CAMERA_CODE){

                    CropImage.activity(immage_uri)
                            .setGuidelines(CropImageView.Guidelines.ON)
                            .start(this);
                    Log.d("immmm",""+immage_uri);
                }
            }
            if (requestCode==CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE){
                CropImage.ActivityResult result=CropImage.getActivityResult(data);
                if (resultCode == RESULT_OK){

                        Uri resultUri = result.getUri();
                        if (btnB){
                            mPreviewIv.setImageURI(resultUri);
                            mPreviewIv.invalidate();
                            BitmapDrawable drawable = (BitmapDrawable) mPreviewIv.getDrawable();
                             bmpimg1 = drawable.getBitmap();
                        }else {
                            mPreviewIv2.setImageURI(resultUri);
                            mPreviewIv2.invalidate();
                            BitmapDrawable drawable = (BitmapDrawable) mPreviewIv2.getDrawable();
                             bmpimg2 = drawable.getBitmap();
                        }




                }
                else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE){
                    Exception error =result.getError();
                    Toast.makeText(this, ""+error, Toast.LENGTH_SHORT).show();
                }
            }
        }





    }