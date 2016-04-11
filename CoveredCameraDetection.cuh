
/*
 *
 *
*/

__device__ bool frameCoveredCamera;

__device__ int findMaxIndex(double * array, int size);
__device__ int SumOfHistogram(double * Diffn, int size);

__global__ void differenceInCurrent_Background(double * In, double * Bn, double * Diffn){
	long int index = blockDim.x*blockIdx.x + threadIdx.x;
	Diffn[index] = abs(In[index] - Bn[index]);
}

__global__ void CoveredCameraDetection(double * H_In, double * H_Bn, float Th1, float Th2, double * Diffn){
	int In_Max = findMaxIndex(H_In, 32);
	int Bn_Max = findMaxIndex(H_Bn, 32);
	int SumofInMax = H_In[In_Max] + H_In[In_Max-1] + H_In[In_Max+1];
	int SumofBnMax = H_Bn[Bn_Max] + H_Bn[Bn_Max-1] + H_Bn[Bn_Max+1];
	int k = 3;
	int Diffn_sum = SumOfHistogram(Diffn, 32);
	int Th_sum = SumOfHistogram(Diffn, k);
	printf("Value of Diffn_sum:%d \t Value of Th_sum:%d\n",Diffn_sum, Th_sum); 
	printf("Max of In:%d \t Max of Bn:%d\n",In_Max, Bn_Max);
	printf("Peak sum In:%d \t Peak sum Bn:%d",SumofInMax, SumofBnMax);
	if ((SumofInMax > Th1*SumofBnMax) && (Diffn_sum > Th2*Th_sum))
		frameCoveredCamera =  true;
	else frameCoveredCamera =  false;
	printf("\n%d\n",frameCoveredCamera);
}

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

__device__ int SumOfHistogram(double * Diffn, int size){
	double Diffn_Sum = 0;
	for(int i =0;i<size;i++)
		Diffn_Sum += Diffn[i];
	return (int)Diffn_Sum;
}