
/* Background subtraction 
 * to detect the non moving 
 * parts in a video sequence
*/

__global__ void FindMovingPixels(double * In, double * In_1, double * In_2,
		double * Tn, double * MovingPixelMap){
	long int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (abs(In[index] - In_1[index])> Tn[index] && abs(In[index] - In_2[index])> Tn[index])
		MovingPixelMap[index] = 255;
	else
		MovingPixelMap[index] = 0;
	__syncthreads();
}

__global__ void UpdateBackgroundImage(double * In, double * MovingPixelMap, double * Bn){
	long int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (MovingPixelMap[index] == 0)
		Bn[index] = 0.92*Bn[index] + 0.08*In[index];

	__syncthreads();
}

__global__ void UpdateThresholdImage( const double * In,double * Tn, 
		 const double * MovingPixelMap, const double * Bn){
	long int index = blockDim.x*blockIdx.x + threadIdx.x;
	int minTh = 20; // Value of minimum threshold for background subtraction
	float th = 0.92 * Tn[index] + 0.24*(abs(In[index] - Bn[index]));
	if (MovingPixelMap[index]==0){
		if (th > minTh)
			Tn[index] = th;
		else Tn[index] = minTh;
	}
	__syncthreads();
}

