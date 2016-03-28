#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    VideoCapture cap("./videos/out.avi"); // open the video file for reading
    
    //VideoCapture cap(0);
    if ( !cap.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the video file" << endl;
         return -1;
    }

    //cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms

    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
	
    double count = cap.get(CV_CAP_PROP_FRAME_COUNT);

    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

    cout << "Frame count : " << count << endl;

    while(1)
    {
        Mat frame;
	cout << "reading the first time" <<endl;
        bool bSuccess = cap.read(frame); // read a new frame from video
	//cap >> frame;
         if (!bSuccess) //if not success, break loop
        {
                       cout << "Cannot read the frame from video file" << endl;
                       //break;
        }
        cvtColor(frame,frame, CV_BGR2GRAY);

        imshow("MyVideo", frame); //show the frame in "MyVideo" window

        cout << frame.data[0] << endl;
	if(waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
        {
                cout << "esc key is pressed by user" << endl; 
                break; 
        }

    }

    return 0;

}
