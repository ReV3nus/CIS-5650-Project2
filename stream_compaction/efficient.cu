#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction
{
    namespace Efficient
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

        // Only paddedN / 2^(d+1) threads exist; each maps to k = index * 2 * stride.
        __global__ void kernUpSweep(int n, int stride, int *data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int k = index * (stride << 1);
            if (k >= n)
            {
                return;
            }
            data[k + (stride << 1) - 1] += data[k + stride - 1];
        }

        __global__ void kernDownSweep(int n, int stride, int *data)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int k = index * (stride << 1);
            if (k >= n)
            {
                return;
            }
            int leftIdx = k + stride - 1;
            int rightIdx = k + (stride << 1) - 1;
            int t = data[leftIdx];
            data[leftIdx] = data[rightIdx];
            data[rightIdx] += t;
        }

        // Baseline for comparison: launches n threads per level and discards idle
        // ones via modulo instead of compacting the thread indices.
        __global__ void kernUpSweepUnoptimized(int n, int stride, int *data)
        {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n)
            {
                return;
            }
            if (k % (stride << 1) == 0)
            {
                data[k + (stride << 1) - 1] += data[k + stride - 1];
            }
        }

        __global__ void kernDownSweepUnoptimized(int n, int stride, int *data)
        {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n)
            {
                return;
            }
            if (k % (stride << 1) == 0)
            {
                int leftIdx = k + stride - 1;
                int rightIdx = k + (stride << 1) - 1;
                int t = data[leftIdx];
                data[leftIdx] = data[rightIdx];
                data[rightIdx] += t;
            }
        }

        // In-place exclusive scan on a power-of-two, zero-padded buffer.
        void devScan(int paddedN, int *dev_data)
        {
            int numLevels = ilog2(paddedN);

            for (int d = 0; d < numLevels; d++)
            {
                int stride = 1 << d;
                int numThreads = paddedN / (stride << 1);
                dim3 blocksPerGrid((numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep<<<blocksPerGrid, BLOCK_SIZE>>>(paddedN, stride, dev_data);
            }

            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));

            for (int d = numLevels - 1; d >= 0; d--)
            {
                int stride = 1 << d;
                int numThreads = paddedN / (stride << 1);
                dim3 blocksPerGrid((numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweep<<<blocksPerGrid, BLOCK_SIZE>>>(paddedN, stride, dev_data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            int paddedN = 1 << ilog2ceil(n);

            int *dev_data;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemset(dev_data, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_data failed!");

            timer().startGpuTimer();
            devScan(paddedN, dev_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_data);
        }

        void scanUnoptimized(int n, int *odata, const int *idata)
        {
            int paddedN = 1 << ilog2ceil(n);

            int *dev_data;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemset(dev_data, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_data failed!");

            int numLevels = ilog2(paddedN);
            dim3 fullBlocksPerGrid((paddedN + BLOCK_SIZE - 1) / BLOCK_SIZE);

            timer().startGpuTimer();

            for (int d = 0; d < numLevels; d++)
            {
                kernUpSweepUnoptimized<<<fullBlocksPerGrid, BLOCK_SIZE>>>(paddedN, 1 << d, dev_data);
            }
            cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));
            for (int d = numLevels - 1; d >= 0; d--)
            {
                kernDownSweepUnoptimized<<<fullBlocksPerGrid, BLOCK_SIZE>>>(paddedN, 1 << d, dev_data);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata)
        {
            int paddedN = 1 << ilog2ceil(n);

            int *dev_idata;
            int *dev_bools;
            int *dev_indices;
            int *dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed!");

            dim3 fullBlocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            timer().startGpuTimer();

            StreamCompaction::Common::kernMapToBoolean<<<fullBlocksPerGrid, BLOCK_SIZE>>>(
                n, dev_bools, dev_idata);

            cudaMemset(dev_indices, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            devScan(paddedN, dev_indices);

            StreamCompaction::Common::kernScatter<<<fullBlocksPerGrid, BLOCK_SIZE>>>(
                n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            int lastBool, lastIndex;
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);

            return count;
        }
    }
}
