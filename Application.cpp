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

bool frameCoveredCamera;

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

void naiveHisto_32 (double *image , double* histo, int width, int height)
{
    for(int i=0;i<width*height;i++){        
        int index = (int) image[i];
        histo[index/8] += 1.0;
    }
}

int findMaxIndex(double * array, int size){
    int maxValue = 0;
    int index = 0;
    for(int i =0;i<size;i++){
        if (array[i] > maxValue){
            maxValue = array[i];
            index = i;
        }
    }
    return index;
}

int SumOfHistogram(double * Diffn, int size){
    double Diffn_Sum = 0;
    for(int i =0;i<size;i++)
        Diffn_Sum += Diffn[i];
    return (int)Diffn_Sum;
}

void differenceInCurrent_Background(double * In, double * Bn, double * Diffn, int width, int height){
    for(int i=0;i<width*height;i++){
        Diffn[i] = abs(In[i] - Bn[i]);
    }
}

void CoveredCameraDetection(double * H_In, double * H_Bn, float Th1, float Th2, double * Diffn){
    int In_Max = findMaxIndex(H_In, 32);
    int Bn_Max = findMaxIndex(H_Bn, 32);
    int SumofInMax = H_In[In_Max] + H_In[In_Max-1] + H_In[In_Max+1];
    int SumofBnMax = H_Bn[Bn_Max] + H_Bn[Bn_Max-1] + H_Bn[Bn_Max+1];
    int k = 3;
    int Diffn_sum = SumOfHistogram(Diffn, 32);
    int Th_sum = SumOfHistogram(Diffn, k);
    // printf("Value of Diffn_sum:%d \t Value of Th_sum:%d\n",Diffn_sum, Th_sum); 
    // printf("Max of In:%d \t Max of Bn:%d\n",In_Max, Bn_Max);
    // printf("Peak sum In:%d \t Peak sum Bn:%d",SumofInMax, SumofBnMax);
    if ((SumofInMax > Th1*SumofBnMax) && (Diffn_sum > Th2*Th_sum))
        frameCoveredCamera =  true;
    else frameCoveredCamera =  false;
    // printf("\n%d\n",frameCoveredCamera);
}
int mem_size;

int main(int argc, char* argv[])
{
    startVideoCapture();
    return 0;
}

void startVideoCapture()
{
    double * In_h, *In_1_h, *In_2_h, *hist_In, *hist_Bn, *hist_Diffn, *Diffn;
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
    Diffn = ( double *)malloc(mem_size*sizeof(double));
    hist_In = ( double *)malloc(32*sizeof(double));
    hist_Bn = ( double *)malloc(32*sizeof(double));
    hist_Diffn = ( double *)malloc(32*sizeof(double));
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

        differenceInCurrent_Background(In_h, Bn_h, Diffn, frame_width, frame_height);

        std::fill_n(hist_In,32, 0);
        std::fill_n(hist_Bn,32, 0);
        std::fill_n(hist_Diffn,32, 0);

        naiveHisto_32(In_h, hist_In, frame_width, frame_height);
        naiveHisto_32(Bn_h, hist_Bn, frame_width, frame_height);
        naiveHisto_32(Diffn, hist_Diffn, frame_width, frame_height);

        CoveredCameraDetection(hist_In, hist_Bn, 1.1, 1.1, hist_Diffn);

        // Mat MovingPixelMap_mat = Mat(frame_height, frame_width, CV_64F, MovingPixelMap_h);
        // Mat In_mat = Mat(frame_height, frame_width, CV_64F, In_h);
        // Mat Tn_mat = Mat(frame_height, frame_width, CV_64F, Tn_h);
        // Mat Bn_mat = Mat(frame_height, frame_width, CV_64F, Bn_h);

        // if (frameCoveredCamera)
        // {
        //     char buffer_1[100];
        //     sprintf(buffer_1, "./output_serial/test_Mp%d.jpg",frame_counter);
        //     saveFrame(MovingPixelMap_mat, buffer_1);
        //     char buffer_2[100];
        //     sprintf(buffer_2, "./output_serial/test_In%d.jpg",frame_counter);
        //     saveFrame(In_mat, buffer_2);
        //     char buffer_3[100];
        //     sprintf(buffer_3, "./output_serial/testTn%d.jpg",frame_counter);
        //     saveFrame(Tn_mat, buffer_3);
        //     char buffer_4[100];
        //     sprintf(buffer_4, "./output_serial/testBn%d.jpg",frame_counter);
        //     saveFrame(Bn_mat, buffer_4);
        // }
        frame_counter++;
    }
}

void saveFrame(Mat frame, char *buff)
{
    imwrite(buff, frame);
}
