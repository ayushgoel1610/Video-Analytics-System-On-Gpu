#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>

/*
 * Real Time video analytics system on GPU
 * Authors:
 * @Ayush Goel - 2012029
 * @Udayan Tandon - 2012167
*/

using namespace cv;

void startVideoCapture();
void saveFrame(Mat frame, char *buff);

/* Background subtraction
* to detect the non moving
* parts in a video sequence
*/

void FindMovingPixels(double In[], double In_1[], double In_2[],
double Tn[], double * MovingPixelMap, int width, int height)
{
    for (int index = 0;
    index <width*height;
    index++)
    {
        if (std::abs(In[index] - In_1[index])> Tn[index] && std::abs(In[index] - In_2[index])> Tn[index])
        MovingPixelMap[index] = 255;
        else
        MovingPixelMap[index] = 0;
    }
}

void UpdateBackgroundImage(double * In, double * MovingPixelMap, double * Bn, int width, int height)
{
    for (int index = 0;
    index <width*height;
    index++)
    {
        if (MovingPixelMap[index] == 0)
        Bn[index] = 0.92*Bn[index] + 0.08*In[index];
    }
}

void UpdateThresholdImage( const double * In,double * Tn,
const double * MovingPixelMap, const double * Bn, int width, int height)
{
    for (int index = 0;
    index <width*height;
    index++)
    {
        int minTh = 20;
        // Value of minimum threshold for background subtraction
        float th = 0.92 * Tn[index] + 0.24*(std::abs(In[index] - Bn[index]));
        if (MovingPixelMap[index]==0)
        {
            if (th > minTh)
            Tn[index] = th;
            else Tn[index] = minTh;
        }
    }
}
int mem_size;

int main(int argc, char* argv[])
{
    startVideoCapture();
    return 0;
}

void startVideoCapture()
{
    double * In_h, *In_1_h, *In_2_h;
    // double * In_d, *In_1_d, *In_2_d;
    double *Tn, *Tn_h, *Bn, *Bn_h, *MovingPixelMap_h;
    int timestamps [] =
    {
        50,150,250,400, 450, 500, 550, 600
    }
    ;
    //Timestamps for milestone 1
    VideoCapture cap("./videos/CameraCovered10_PartiallyThenWhole_25.avi");
    // open the video file for reading
    if ( !cap.isOpened() )  // if not success, exit program
    {
        std::cout << "Cannot open the video file" << std::endl;
        return ;
    }
    long frame_count = cap.get(CV_CAP_PROP_FRAME_COUNT);
    long frame_counter = 0;
    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    mem_size = frame_height*frame_width;
    std::cout << frame_height << " " << frame_width <<  " " << frame_count << std::endl;
    In_h = ( double *)malloc(mem_size*sizeof(double));
    In_1_h = ( double *)malloc(mem_size*sizeof(double));
    In_2_h = ( double *)malloc(mem_size*sizeof(double));
    MovingPixelMap_h = ( double *)malloc(mem_size*sizeof(double));
    Tn_h = ( double *)malloc(mem_size*sizeof(double));
    Bn_h = ( double *)malloc(mem_size*sizeof(double));
    std::fill_n(Bn_h, frame_height*frame_width, 0);
    std::fill_n(Tn_h, frame_height*frame_width , 127);
    double * p;
    while(frame_counter < frame_count)
    {
        Mat frame;
        bool success = cap.read(frame);
        if (!success) //if not success, break loop
        {
            std::cout << "Cannot read the frame  " << std::endl;
            break;
        }
        cvtColor(frame,frame, CV_BGR2GRAY);
        frame.convertTo(frame, CV_64F);
        memcpy( In_2_h, In_1_h, mem_size*sizeof(double));
        memcpy(In_1_h, In_h, mem_size*sizeof(double));
        for (int i = 0;
        i < frame_height;
        ++i)
        {
            p = frame.ptr<double>(i);
            for (int j = 0;
            j < frame_width;
            ++j)
            {
                In_h[i * frame_width + j] = (double) p[j];
            }
        }
        if(frame_counter<3)
        {
            frame_counter++;
            continue;
        }
        FindMovingPixels(In_h, In_1_h, In_2_h, Tn_h, MovingPixelMap_h, frame_width, frame_height);
        UpdateBackgroundImage(In_h, MovingPixelMap_h, Bn_h, frame_width, frame_height);
        UpdateThresholdImage(In_h, Tn_h, MovingPixelMap_h, Bn_h,frame_width, frame_height);

        Mat MovingPixelMap_mat = Mat(frame_height, frame_width, CV_64F, MovingPixelMap_h);
        Mat In_mat = Mat(frame_height, frame_width, CV_64F, In_h);
        Mat Tn_mat = Mat(frame_height, frame_width, CV_64F, Tn_h);
        Mat Bn_mat = Mat(frame_height, frame_width, CV_64F, Bn_h);

        if (frame_counter%10==0)
        {
            char buffer_1[100];
            sprintf(buffer_1, "./output/test_Mp%d.jpg",frame_counter);
            saveFrame(MovingPixelMap_mat, buffer_1);
            char buffer_2[100];
            sprintf(buffer_2, "./output/test_In%d.jpg",frame_counter);
            saveFrame(In_mat, buffer_2);
            char buffer_3[100];
            sprintf(buffer_3, "./output/testTn%d.jpg",frame_counter);
            saveFrame(Tn_mat, buffer_3);
            char buffer_4[100];
            sprintf(buffer_4, "./output/testBn%d.jpg",frame_counter);
            saveFrame(Bn_mat, buffer_4);
        }
        frame_counter++;
    }
}

void saveFrame(Mat frame, char *buff)
{
    imwrite(buff, frame);
}
