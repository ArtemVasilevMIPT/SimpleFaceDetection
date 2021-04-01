#include "opencv2/objdetect.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;

CascadeClassifier faceCascade;

Mat detectAndDraw(const Mat& mt)
{

    Mat copy = mt.clone();
    Mat gray;
    cvtColor(copy, gray, COLOR_BGR2GRAY);
    resize(gray, gray, Size(300, 300));
    resize(copy, copy, Size(300, 300));

    std::vector<Rect> faces;
    faceCascade.detectMultiScale(gray, faces);

    for(const auto& face : faces)
    {
         rectangle(copy, face, COLOR_BGR2BGR555);
    }

    resize(copy, copy, Size(mt.cols, mt.rows));
    return copy;
}

int main()
{
    //Startup message
    std::cout << "This is a simple face detection program. It uses Haar Cascade."
                 << std::endl << "In order to quit press escape" << std::endl;

    //Loading Haar Cascade Classifier and applying it to image
    std::string faceCascadePath = "../Classifiers/haarcascade_frontalface_alt.xml";

    faceCascade.load(faceCascadePath);
    VideoCapture capture;
    capture.open(0);
    if(!capture.isOpened())
    {
        std::cout << "Cannot open camera device" << std::endl;
        return 2;
    }
    Mat frame;

    while(capture.read(frame))
    {
        if(frame.empty())
        {
            std::cout << "No frame to read" << std::endl;
            break;
        }

        Mat output = detectAndDraw(frame);
        imshow("Output", output);

        if(waitKey(10) == 27)
        {
            break; //escape
        }
    }
    return 0;
}
