
/*
 *
 *
*/

__device__ bool frameCoveredCamera;

__device__ int findMaxIndex(float * array, int size);
__device__ int SumOfHistogram(double * Diffn, int size);

__global__ void differenceInCurrent_Background(double * In, double * Bn, double * Diffn){
	long int index = blockDim.x*blockIdx.x + threadIdx.x;
	Diffn[index] = abs(In[index] - Bn[index]);
}

__global__ void CoveredCameraDetection(float * H_In, float * H_Bn, int Th1, int Th2, double * Diffn){
	int In_Max = findMaxIndex(H_In, 32);
	int Bn_Max = findMaxIndex(H_Bn, 32);
	int k = 0;
	int Diffn_sum = SumOfHistogram(Diffn, 32);
	int Th_sum = SumOfHistogram(Diffn, k);
	if ((H_In[In_Max] + H_In[In_Max-1] + H_In[In_Max+1] > Th1*(H_Bn[Bn_Max] + H_Bn[Bn_Max-1] + H_Bn[Bn_Max+1])) && (Diffn_sum > Th2*(Th_sum)))
		frameCoveredCamera =  true;
	frameCoveredCamera =  false;
}

__device__ int findMaxIndex(float * array, int size){
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
	float Diffn_Sum = 0;
	for(int i =0;i<size;i++)
		Diffn_Sum += Diffn[i];
	return (int)Diffn_Sum;
}