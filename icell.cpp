
///#include <linux/videodev2.h>


///#include <iostream>



///#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui_c.h>
///#include <opencv2/highgui.hpp>
///#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
///#include <opencv2/features2d/features2d.hpp>
///#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
///#include <opencv2/flann.hpp>


#include "kbhit.h"


using namespace std;
using namespace cv;



char ctrlr = '.';
char fn[1024];


static void onMouse(int evt, int x, int y, int flags, void* param) {
    if(evt == CV_EVENT_LBUTTONDOWN) {
        cv::Point* ptPtr = (cv::Point*)param;
        ptPtr->x = x;
        ptPtr->y = y;
    }
}

/// DETECTOR TYPES  ---------------------------------
//   1 - cv::AgastFeatureDetector
//   2 - cv::AKAZE
//   3 - cv::BRISK
//   4 - cv::FastFeatureDetector
//   5 - cv::GFTTDDetector
//   6 - cv::KAZE
//   7 - cv::MSER
//   8 - cv::ORB
//   9 - cv::SimpleBlobDetector
//  10 - cv::xfeatures2d::AffineFeature2D
//  11 - cv::xfeatures2d::BoostDesc
//  12 - cv::xfeatures2d::BriefDescriptorExtractor
//  13 - cv::xfeatures2d::DAISY
//  14 - cv::xfeatures2d::FREAK
//  15 - cv::xfeatures2d::HarrisLaplaceFeatureDetector
//  16 - cv::xfeatures2d::LATCH
//  17 - cv::xfeatures2d::LUCID
//  18 - cv::xfeatures2d::MSDDetector
//  19 - cv::xfeatures2d::SIFT
//  20 - cv::xfeatures2d::StarDetector
//  21 - cv::xfeatures2d::SURF
//  22 - cv::xfeatures2d::VGG
/// ---------------------------------------------------

#define DETECTOR_TYPE   2

static Ptr<Feature2D> CreateDetector(){
#if(DETECTOR_TYPE == 1)
    return cv::AgastFeatureDetector::create();                      ///n
#elif(DETECTOR_TYPE == 2)
    return cv::AKAZE::create();                                     ///ok
#elif(DETECTOR_TYPE == 3)
    return cv::BRISK::create();                                     ///ok
#elif(DETECTOR_TYPE == 4)
    return cv::FastFeatureDetector::create();
#elif(DETECTOR_TYPE == 5)
    return cv::GFTTDetector::create();
#elif(DETECTOR_TYPE == 6)
    return cv::KAZE::create();
#elif(DETECTOR_TYPE == 7)
    return cv::MSER::create();
#elif(DETECTOR_TYPE == 8)
    return cv::ORB::create();
#elif(DETECTOR_TYPE == 9)
    return cv::SimpleBlobDetector::create();                        ///n
#elif(DETECTOR_TYPE == 10)
    return xfeatures2d::AffineFeature2D::create();                  ///n
#elif(DETECTOR_TYPE == 11)
    return xfeatures2d::BoostDesc::create();
#elif(DETECTOR_TYPE == 12)
    return xfeatures2d::BriefDescriptorExtractor::create();
#elif(DETECTOR_TYPE == 13)
    return xfeatures2d::DAISY::create();
#elif(DETECTOR_TYPE == 14)
    return xfeatures2d::FREAK::create();
#elif(DETECTOR_TYPE == 15)
    return xfeatures2d::HarrisLaplaceFeatureDetector::create();
#elif(DETECTOR_TYPE == 16)
    return xfeatures2d::LATCH::create();
#elif(DETECTOR_TYPE == 17)
    return xfeatures2d::LUCID::create();
#elif(DETECTOR_TYPE == 18)
    return xfeatures2d::MSDDetector::create();                      ///n
#elif(DETECTOR_TYPE == 19)
    return xfeatures2d::SiftFeatureDetector::create();              ///ok
#elif(DETECTOR_TYPE == 20)
    return xfeatures2d::StarDetector::create();                     ///n
#elif(DETECTOR_TYPE == 21)
    return xfeatures2d::SurfFeatureDetector::create();              ///n
#elif(DETECTOR_TYPE == 22)
    return xfeatures2d::VGG::create();                              ///n
#endif



}

int main(){

    Mat frame;
    Mat img_matches;
    Mat imCrop1;
    Mat imCrop2;
    Mat imRef;

    Rect2d r;
    Rect2d rRef;
    Point2i pt(-1,-1);//assume initial point
    Point2i pf,ps;


    FILE *f = popen("zenity --file-selection", "r"); // popen starts a new process. zenity is a process that creates a dialog box
    fgets(fn, 1024, f);
    /// There is a '/n' character in the end of string
    int sz = strlen(fn);
    if(sz == 0){
        return -1;
    }
    fn[sz-1] = 0;

    VideoCapture cap(fn);

    if(!cap.open(fn)){
        return 0;
    }

    namedWindow("view", CV_WINDOW_AUTOSIZE); // Create a window
///    namedWindow("edges", CV_WINDOW_AUTOSIZE); // Create a window
    namedWindow("matches", CV_WINDOW_AUTOSIZE);
    namedWindow("matchesRef", CV_WINDOW_AUTOSIZE);
    startWindowThread();


    setMouseCallback("view", onMouse, (void*)&pt);


    /// Select ROI from first image

    cap >> frame;


    /// Select ROI
    r = selectROI(frame);
    rRef = r;


    /// Crop image
    imCrop1 = frame.clone();
    imCrop1 = imCrop1(r);

    /// Create structures
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    /// Create detector
    Ptr<Feature2D> detectorF2D = CreateDetector();

    /// Extract key points from 1st image
    detectorF2D->detect(imCrop1, keypoints1);

    /// Computing descriptors
    detectorF2D->compute(imCrop1, keypoints1, descriptors1);




    while (ctrlr != 'q'){
        while(_kbhit()){
            ctrlr = getch();
        }
        cap >> frame;
        if(frame.empty()){
            break;
        }

        /// Tracking
        imCrop2 = frame.clone();
        imCrop2 = imCrop2(r);

        imRef = frame(rRef);
        ///imRef = frame.clone();
        ///imRef = imRef(r);

        /// Show Images
        imshow("matchesRef", imRef);
        imshow("view", frame);
        waitKey(1);


        /// Extract key points from 2nd image
        detectorF2D->detect(imCrop2, keypoints2);

        /// Computing descriptors
        detectorF2D->compute(imCrop2, keypoints2, descriptors2);



        /// matching descriptors
        vector<DMatch> matches, matchesGMS;
        BFMatcher matcher(NORM_L2);
        //FlannBasedMatcher matcher;
        matcher.match(descriptors1, descriptors2, matches);



        /// drawing the results


        cv::xfeatures2d::matchGMS(imCrop1.size(), imCrop2.size(), keypoints1, keypoints2, matches, matchesGMS, 1 /*withRotation*/, 1 /*withScale*/);

        drawMatches(imCrop1, keypoints1, imCrop2, keypoints2, matchesGMS, img_matches);

        ///drawKeypoints(imCrop1,keypoints1,img_matches);



        imshow("matches", img_matches);

        waitKey(1);



        /// queryIdx -> keypoints1 - primeira imagem
        /// trainIdx -> keypoints2 - segunda imagem



        ///Update rectangle
        int szmc = matchesGMS.size();
        ///int szmc = matches.size();


        float xm = 0;
        float ym = 0;

        if(szmc){
            for(int i=0;i<szmc;i++){
              xm = xm + (keypoints2.at(matchesGMS.at(i).trainIdx).pt.x - keypoints1.at(matchesGMS.at(i).queryIdx).pt.x);
              ym = ym + (keypoints2.at(matchesGMS.at(i).trainIdx).pt.y - keypoints1.at(matchesGMS.at(i).queryIdx).pt.y);

              ///xm = xm + (keypoints2.at(matches.at(i).trainIdx).pt.x - keypoints1.at(matches.at(i).queryIdx).pt.x);
              ///ym = ym + (keypoints2.at(matches.at(i).trainIdx).pt.y - keypoints1.at(matches.at(i).queryIdx).pt.y);

            }
            xm = xm/szmc;
            ym = ym/szmc;
        }else{


        }


        r.x = r.x + xm;
        r.y = r.y + ym;


        imCrop1 = imCrop2.clone();
        keypoints1 = keypoints2;
        descriptors1 = descriptors2;

/*
        /// Test of edges detector
        cvtColor(frmN, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, temp, Size(7,7), 1.5, 1.5);
        Canny(temp, edges, 0, 30, 3);
*/


    }


    cap.release();
    destroyAllWindows();

    return 0;
}
