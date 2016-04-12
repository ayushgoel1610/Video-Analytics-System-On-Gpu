
/* Covered camera detection algorithms
 *
*/

//Need this variable inside host, therefore declared as host variable
// __device__ bool frameCoveredCamera;

__device__ int findMaxIndex(double * array, int size);
__device__ int SumOfHistogram(double * Diffn, int size);

//Computes difference between current and background frame
__global__ void differenceInCurrent_Background(double * In, double * Bn, double * Diffn){
	long int index = blockDim.x*blockIdx.x + threadIdx.x;
	Diffn[index] = abs(In[index] - Bn[index]);
}

//Computes the value of the two equations required to be satisfied in order to detect a covered camera frame. 
__global__ void CoveredCameraDetection(double * H_In, double * H_Bn, float Th1, float Th2, double * Diffn, bool * frameCoveredCamera){
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
		*frameCoveredCamera =  true;
	else *frameCoveredCamera =  false;
	// printf("\n%d\n",frameCoveredCamera);
}

/*
 * Finds the bin number corresponding 
 * to maximum value in the histogram
*/
__device__ int findMaxIndex(double * array, int size){
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

/*
 * Computes the sum of the histogram for
 * a given range.
*/
__device__ int SumOfHistogram(double * Diffn, int size){
	double Diffn_Sum = 0;
	for(int i =0;i<size;i++)
		Diffn_Sum += Diffn[i];
	return (int)Diffn_Sum;
}