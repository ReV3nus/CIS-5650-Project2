#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction
{
    namespace Naive
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        #ifndef BLOCK_SIZE
        #define BLOCK_SIZE 128
        #endif

        // Hillis-Steele inclusive scan, one pass per level.
        __global__ void kernNaiveScanStep(int n, int offset, int *odata, const int *idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            if (index >= offset)
            {
                odata[index] = idata[index] + idata[index - offset];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        // Shift inclusive scan right by one to make it exclusive.
        __global__ void kernInclusiveToExclusive(int n, int *odata, const int *idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            odata[index] = index == 0 ? 0 : idata[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            int *dev_bufA;
            int *dev_bufB;
            cudaMalloc((void**)&dev_bufA, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufA failed!");
            cudaMalloc((void**)&dev_bufB, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufB failed!");

            cudaMemcpy(dev_bufA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_bufA failed!");

            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            timer().startGpuTimer();

            int *in = dev_bufA;
            int *out = dev_bufB;
            int numSteps = ilog2ceil(n);
            for (int d = 1; d <= numSteps; d++)
            {
                int offset = 1 << (d - 1);
                kernNaiveScanStep<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, offset, out, in);
                std::swap(in, out);
            }
            kernInclusiveToExclusive<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, out, in);

            timer().endGpuTimer();

            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_bufA);
            cudaFree(dev_bufB);
        }
    }
}
