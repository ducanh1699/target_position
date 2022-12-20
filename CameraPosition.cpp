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

int main()
{
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

    // marker length: 0.75m
    float markerlength = 0.75;

    string image_path = "2.jpg";
    cv::Mat markerImage = cv::imread(image_path);
    // cout << markerImage << endl;
    cv::aruco::PREDEFINED_DICTIONARY_NAME dictionary_;
    dictionary_ = cv::aruco::DICT_4X4_1000;
    // Load the dictionary that was used to generate the markers.
    Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(dictionary_);
    Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    // Declare the vectors that would contain the detected marker corners and the rejected marker candidates
    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    // The ids of the detected markers are stored in a vector
    vector<int> markerIds;

    // corners of each marker in the detected image
    vector<vector<Point2f>> markerCorner;
    // id of each marker in the detected image
    int markerId;

    // pose of camera in marker frame
    cv::Mat rvec = Mat(3,1,CV_64FC1);
    cv::Mat tvec = Mat(3,1,CV_64FC1);
    cv::Mat Rotation = Mat(3,3, CV_64FC1);
    cv::Mat rvec1 = Mat(3, 1, CV_64FC1);
    cv::Mat tvec1 = Mat(3, 1, CV_64FC1);

    // input for camera matrix D435
    // float minusmatrix_array[3][3] = {{-1,0,0},{0,-1,0},(0,0,-1)};
    cv::Mat_<double> minusmatrix = Mat(3, 3, CV_64FC1);
    minusmatrix << -1.0, 0, 0,
                    0, -1.0, 0,
                    0, 0, -1.0;

    // minusmatrix.at<double>(0, 0) = -1.0;
    // minusmatrix.at<double>(0, 1) = 0.0;
    // minusmatrix.at<double>(0, 2) = 0.0;
    // minusmatrix.at<double>(1, 0) = 0.0;
    // minusmatrix.at<double>(1, 1) = -1.0;
    // minusmatrix.at<double>(1, 2) = 0.0;
    // minusmatrix.at<double>(2, 0) = 0.0;
    // minusmatrix.at<double>(2, 1) = 0.0;
    // minusmatrix.at<double>(2, 2) = -1.0;
    // Position of camera in marker frame
    cv::Mat tvec_marker = Mat(3, 1, CV_64FC1);

    // vector<Point3f> rvec1;

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
            cout << markerIds[0] << endl;
            cout << tvec.at<double>(0) << " " << tvec.at<double>(1) << " " << tvec.at<double>(2) << endl;
            // cout << rvec.at<double>(0) << " " << rvec.at<double>(1) << " " << rvec.at<double>(2) << endl;
            cout<<"Tvec: "<<tvec<<endl;
            // rvec1.at<double>(0) = rvec.at<double>(0);
            // rvec1.at<double>(1) = rvec.at<double>(1);
            // rvec1.at<double>(2) = rvec.at<double>(2);
            tvec1.at<double>(0) = tvec.at<double>(0);
            tvec1.at<double>(1) = tvec.at<double>(1);
            tvec1.at<double>(2) = tvec.at<double>(2);
            // tvec1 = tvec;
            cout << tvec1.at<double>(0) << " " << tvec1.at<double>(1) << " " << tvec1.at<double>(2) << endl;

            cout<<"Tvec1: "<<tvec1<<endl;

            //tvec1 khac tvec???

            int m1 = tvec1.rows;
            int m2 = tvec1.cols;

            cout << m1 << " " << m2 << endl;

            
            cv::Rodrigues(rvec1, Rotation);
            cout << Rotation << endl;
            Rotation = Rotation*-1.0;
            cout << Rotation << endl;
            // int n1 = Rotation.rows;
            // int n2 = Rotation.cols;

            // cout << n1 << " " << n2 << endl;
            // cout << Rotation.at<double>(2,0) << " " << Rotation.at<double>(2,1) << " " << Rotation.at<double>(2,2) << endl;
            // cout << "Rotation: " << endl
            //      << Rotation << endl <<"Minus Matrix: "<<minusmatrix<<endl;
            
            // Rotation = Rotation * minusmatrix;
            // for (int i = 0; i < 3; i++)
            // {
            //     for (int j = 0; i < 3; j++)
            //     {
            //         Rotation.at<double>(i, j) = Rotation.at<double>(i, j)
            //     }
            // }

            Rotation = Rotation.t();
            cout << "Rotation: " << endl
                 << Rotation << endl;
            cout << "tvec1: "<<tvec1<<endl;
            tvec_marker = Rotation*tvec1;
            // int p1 = tvec_marker.rows;
            // cout<<"---"<< p1<<"-----"<<endl;
            cout << "Camera in marker frame: " << tvec_marker << endl;
            cv::waitKey(0);
        }
    }
    cv::waitKey(0);
    return 0;
}
