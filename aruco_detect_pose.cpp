#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <fstream>
#include <string.h>

using namespace std;
using namespace Eigen;
using namespace cv;

int main(){    
    // input for camera matrix D435
    cv::Mat camera_matrix = Mat(3, 3, CV_64FC1);

    camera_matrix.at<double>(0, 0) = 917.4971923828125;
    camera_matrix.at<double>(0, 1) = 0.0;
    camera_matrix.at<double>(0, 2) = 635.0016479492188;
    camera_matrix.at<double>(1, 0) = 0.0;
    camera_matrix.at<double>(1, 1) = 915.86474609375;
    camera_matrix.at<double>(1, 2) = 368.91497802734375;
    camera_matrix.at<double>(2, 0) = 0.0;
    camera_matrix.at<double>(2, 1) = 0.0;
    camera_matrix.at<double>(2, 2) = 1.0;

    cv::Mat camera_dist = cv::Mat::zeros(1, 5, CV_64FC1);
    // = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    //marker length: 0.75m
    float markerlength = 0.75;

    string image_path = "2.jpg";
    cv::Mat markerImage = cv::imread(image_path);
    // cout << markerImage << endl;

    // Load the dictionary that was used to generate the markers.
    Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_1000);
    Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    // Declare the vectors that would contain the detected marker corners and the rejected marker candidates
    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    // The ids of the detected markers are stored in a vector
    vector<int> markerIds;

    // corners of each marker in the detected image
    vector<vector<Point2f>> markerCorner;
    // id of each marker in the detected image
    int markerId;

    // pose of marker in camera frame
    cv::Mat rvec, tvec;
    cv::Mat Rotation;

    // Position of camera in marker frame
    cv::Mat tvec_marker;

    // Detect the markers in the image
    cv::aruco::detectMarkers(markerImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
    if (markerIds.size() > 0)
    {
        // each marker in the image
        for (int i = 0; i < markerIds.size(); i++)
        {
            markerId = markerIds[i];
            markerCorner[0] = markerCorner[i];
            cv::aruco::estimatePoseSingleMarkers(markerCorners, markerlength, camera_matrix, camera_dist, rvec, tvec);

            //(15/12) FAIL!

            // cout << tvec.at<double>(0) << endl;
            // cout << tvec.at<double>(1) << endl;
            // cout << tvec.at<double>(2) << endl;
            // Rodrigues(rvec, Rotation);
            // tvec_marker = (-Rotation.t()) * tvec;
            cout<<tvec.at<double>(0)<<" "<<tvec.at<double>(1)<<" "<<tvec.at<double>(2)<<endl;
            cv::waitKey(0);
        }
    }
    cv::waitKey(0);
    return 0;
}