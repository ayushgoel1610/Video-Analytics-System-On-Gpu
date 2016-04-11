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


__global__ void
naiveHistoKernel_32 (double *image , double* histo)
{
      int id = blockIdx.x * blockDim.x + threadIdx.x ;
      int index = (int) image[id];
      atomicAdd (&histo[index/8] , 1.0 );

}
