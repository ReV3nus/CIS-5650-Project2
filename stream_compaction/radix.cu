#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix.h"

namespace StreamCompaction
{
    namespace Radix
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

        // e[i] = 1 when bit `bit` of idata[i] is 0.
        __global__ void kernComputeE(int n, int bit, int *e, const int *idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            e[index] = ((idata[index] >> bit) & 1) ^ 1;
        }

        // totalFalses = e[n-1] + f[n-1], computed on-device to avoid a host sync per pass.
        __global__ void kernComputeTotalFalses(int n, int *totalFalses,
                const int *e, const int *f)
        {
            *totalFalses = e[n - 1] + f[n - 1];
        }

        __global__ void kernSplit(int n, int *odata, const int *idata,
                const int *e, const int *f, const int *totalFalses)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            int destination = e[index] ? f[index]
                                       : index - f[index] + *totalFalses;
            odata[destination] = idata[index];
        }

        // LSD radix sort using StreamCompaction::Efficient::devScan for the per-bit split.
        // Non-negative ints only.
        void sort(int n, int *odata, const int *idata)
        {
            int maxVal = 0;
            for (int i = 0; i < n; i++)
            {
                maxVal = std::max(maxVal, idata[i]);
            }
            int numBits = ilog2ceil(maxVal + 1);

            int paddedN = 1 << ilog2ceil(n);

            int *dev_bufA;
            int *dev_bufB;
            int *dev_e;
            int *dev_f;
            int *dev_totalFalses;

            cudaMalloc((void**)&dev_bufA, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufA failed!");
            cudaMalloc((void**)&dev_bufB, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufB failed!");
            cudaMalloc((void**)&dev_e, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_e failed!");
            cudaMalloc((void**)&dev_f, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_f failed!");
            cudaMalloc((void**)&dev_totalFalses, sizeof(int));
            checkCUDAError("cudaMalloc dev_totalFalses failed!");

            cudaMemcpy(dev_bufA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_bufA failed!");

            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            timer().startGpuTimer();

            int *in = dev_bufA;
            int *out = dev_bufB;
            for (int bit = 0; bit < numBits; bit++)
            {
                kernComputeE<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, bit, dev_e, in);

                cudaMemset(dev_f, 0, paddedN * sizeof(int));
                cudaMemcpy(dev_f, dev_e, n * sizeof(int), cudaMemcpyDeviceToDevice);
                StreamCompaction::Efficient::devScan(paddedN, dev_f);

                kernComputeTotalFalses<<<1, 1>>>(n, dev_totalFalses, dev_e, dev_f);
                kernSplit<<<fullBlocksPerGrid, BLOCK_SIZE>>>(n, out, in, dev_e, dev_f, dev_totalFalses);

                std::swap(in, out);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, in, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_bufA);
            cudaFree(dev_bufB);
            cudaFree(dev_e);
            cudaFree(dev_f);
            cudaFree(dev_totalFalses);
        }
    }
}
