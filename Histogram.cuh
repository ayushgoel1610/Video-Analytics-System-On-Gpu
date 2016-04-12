
/*
 * Functions required to compute histograms 
 * which in turn are required for covered camera
 * algorithms.
*/

 /*
  * Initialize kernel to given value. Preferred this over
  * CudaMemSet because cudamemset doesn't work on float/double
  * data types
*/
template<typename T>
__global__ void initKernel(T * devPtr, const T val, const size_t nwords)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    devPtr[tidx] = val;
}

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/*
 * Convert given image into a 32 bin
 * histogram
*/ 

__global__ void
naiveHistoKernel_32 (double *image , double* histo)
{	
	__shared__ double temp[32];
	if(threadIdx.x<32){
     	temp[threadIdx.x] = 0;
     	__syncthreads();
     }
    int id = blockIdx.x * blockDim.x + threadIdx.x ;
    int index = (int) image[id];
    atomicAdd (&temp[index/8] , 1.0);
    __syncthreads();
    if(threadIdx.x<32){
    	atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
    }

}
