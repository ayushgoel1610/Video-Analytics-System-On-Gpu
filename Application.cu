#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "BackgroundSubtraction.cuh"
#include "Histogram.cuh"
#include "CoveredCameraDetection.cuh"

/*
 * Real Time video analytics system on GPU
 * Authors:
 * @Ayush Goel - 2012029
 * @Udayan Tandon - 2012167
*/

using namespace cv;

void startVideoCapture();
void saveFrame(Mat frame, char *buff);
void printArr(double *arr, int size);

int mem_size;

int main(int argc, char* argv[])
{
    startVideoCapture();
    return 0;
}

void startVideoCapture(){
    // Host Image array
    double * In_h;

    /*
     * Device Image arrays
     * Three buffers used 
     */   
    double * In_d, *In_1_d, *In_2_d, * Diffn_d;

    /*
     * Threshold, Background and MovingPixelMap image arrays
    */
    double *Tn_d, *Tn_h, *Bn_d, *Bn_h, *MovingPixelMap_d, *MovingPixelMap_h, *hist_In_d, *hist_h, *hist_Bn_d, *hist_Diffn_d;

    VideoCapture cap("./videos/CameraCovered10_PartiallyThenWhole_25.avi"); // open the video file for reading
    
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
    MovingPixelMap_h = ( double *)malloc(mem_size*sizeof(double));
    Tn_h = ( double *)malloc(mem_size*sizeof(double));
    Bn_h = ( double *)malloc(mem_size*sizeof(double));
    //hist = (double *)malloc(32*sizeof(double));

    double * p;

    cudaMalloc((void **)&In_d, mem_size*sizeof(double));
    cudaMalloc((void **)&In_1_d, mem_size*sizeof(double));
    cudaMalloc((void **)&In_2_d, mem_size*sizeof(double));
    cudaMalloc((void **)&Bn_d, mem_size*sizeof(double));
    cudaMalloc((void **)&Tn_d, mem_size*sizeof(double));
    cudaMalloc((void **)&Diffn_d, mem_size*sizeof(double));
    cudaMalloc((void **)&MovingPixelMap_d, mem_size*sizeof(double));

    cudaMalloc((void **)&hist_In_d, 32*sizeof(double));
    cudaMalloc((void **)&hist_Bn_d, 32*sizeof(double));
    cudaMalloc((void **)&hist_Diffn_d, 32*sizeof(double));

    std::fill_n(Tn_h, frame_height*frame_width, 127);
    std::fill_n(Bn_h, frame_height*frame_width, 0);

    cudaMemcpy(Bn_d, Bn_h , mem_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Tn_d, Tn_h , mem_size*sizeof(double), cudaMemcpyHostToDevice);

    while(frame_counter < frame_count)
    {
        Mat frame;

        bool success = cap.read(frame);

        if (!success) //if not success, break loop
        {
           std::cout << "Cannot read the frame  " << std::endl;
           break;
        }

        //Convert frame to single channel
        cvtColor(frame,frame, CV_BGR2GRAY);

        frame.convertTo(frame, CV_64F);

        /*
         * Put the frame into the image array
        */
        for (int i = 0; i < frame_height; ++i) {
            p = frame.ptr<double>(i);
            for (int j = 0; j < frame_width; ++j) {
                In_h[i * frame_width + j] = (double) p[j];
            }
        }

        // Swapping buffers everytime you get a new frame
    	cudaMemcpy( In_2_d, In_1_d, mem_size*sizeof(double), cudaMemcpyDeviceToDevice);
    	cudaMemcpy( In_1_d, In_d, mem_size*sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy( In_d, In_h, mem_size*sizeof(double), cudaMemcpyHostToDevice);
            
        /*
         * For the first three iterations
         * there are no previous and previous to previous 
         * current frames, hence skipped
        */    
    	if(frame_counter<3){
            frame_counter++;
    		continue;
    	}

        // Kernel for computing the Moving Pixel Map
        FindMovingPixels<<<frame_height, frame_width>>>(In_d, In_1_d, In_2_d, Tn_d, MovingPixelMap_d);

        // Kernel for updating the background image
	    UpdateBackgroundImage<<<frame_height, frame_width>>>(In_d, MovingPixelMap_d, Bn_d);

        // Kernel for updating the threshold image
	    UpdateThresholdImage<<<frame_height, frame_width>>>(In_d, Tn_d, MovingPixelMap_d, Bn_d);

        std::fill_n(hist_h, 32, 0);
        //cudaMemcpy(hist_d, hist_h, 32*sizeof(double), cudaMemcpyHostToDevice);

        differenceInCurrent_Background<<<frame_height, frame_width>>>(In_d, Bn_d, Diffn_d);

        naiveHistoKernel_32<<<frame_height,frame_width>>>(In_d, hist_In_d);
        naiveHistoKernel_32<<<frame_height,frame_width>>>(Bn_d, hist_Bn_d);
        naiveHistoKernel_32<<<frame_height,frame_width>>>(Diffn_d, hist_Diffn_d);
        
        //Copying images from device to host
        cudaMemcpy( MovingPixelMap_h, MovingPixelMap_d, mem_size*sizeof(double), cudaMemcpyDeviceToHost);
	    cudaMemcpy(Bn_h, Bn_d, mem_size*sizeof(double), cudaMemcpyDeviceToHost);
	    cudaMemcpy(Tn_h, Tn_d, mem_size*sizeof(double), cudaMemcpyDeviceToHost);
        //cudaMemcpy(hist_h, hist_d, 32*sizeof(double), cudaMemcpyDeviceToHost);
        //printArr(hist_h, 32);

        // Converting to opencv::Mat in order to save it as jpeg file
        Mat MovingPixelMap_mat = Mat(frame_height, frame_width, CV_64F, MovingPixelMap_h);
	    Mat In_mat = Mat(frame_height, frame_width, CV_64F, In_h);
        Mat Tn_mat = Mat(frame_height, frame_width, CV_64F, Tn_h);
	    Mat Bn_mat = Mat(frame_height, frame_width, CV_64F, Bn_h);
        
        if (frame_counter%10==0){
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

void saveFrame(Mat frame, char *buff){
    imwrite(buff, frame);
}

void printArr(double *arr, int size){
    int i=0;
    for(i=0;i<size;i++){
        printf("%G    ", arr[i]);
    }
    printf("\n");
}

