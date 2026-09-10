#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "sharedmem.h"

namespace StreamCompaction
{
    namespace SharedMemory
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

        #define elemsPerBlock (BLOCK_SIZE * 2)

        // 32 shared-memory banks; pad one word every 32 to dodge conflicts.
        #define LOG_NUM_BANKS 5
        #define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)

        // GPU Gems 3 Example 39-2: full block scan in shared memory.
        // Writes the block's total to blockSums[blockIdx.x] before zeroing the root, if non-null.
        __global__ void kernScanBlockShared(int n, int *odata, const int *idata,
                int *blockSums)
        {
            extern __shared__ int temp[];

            int thid = threadIdx.x;
            int blockOffset = blockIdx.x * elemsPerBlock;

            int ai = thid;
            int bi = thid + blockDim.x;
            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            temp[ai + bankOffsetA] =
                (blockOffset + ai < n) ? idata[blockOffset + ai] : 0;
            temp[bi + bankOffsetB] =
                (blockOffset + bi < n) ? idata[blockOffset + bi] : 0;

            int offset = 1;

            for (int d = elemsPerBlock >> 1; d > 0; d >>= 1)
            {
                __syncthreads();
                if (thid < d)
                {
                    int l = offset * (2 * thid + 1) - 1;
                    int r = offset * (2 * thid + 2) - 1;
                    l += CONFLICT_FREE_OFFSET(l);
                    r += CONFLICT_FREE_OFFSET(r);
                    temp[r] += temp[l];
                }
                offset <<= 1;
            }

            if (thid == 0)
            {
                int root = elemsPerBlock - 1;
                root += CONFLICT_FREE_OFFSET(root);
                if (blockSums != nullptr)
                {
                    blockSums[blockIdx.x] = temp[root];
                }
                temp[root] = 0;
            }

            for (int d = 1; d < elemsPerBlock; d <<= 1)
            {
                offset >>= 1;
                __syncthreads();
                if (thid < d)
                {
                    int l = offset * (2 * thid + 1) - 1;
                    int r = offset * (2 * thid + 2) - 1;
                    l += CONFLICT_FREE_OFFSET(l);
                    r += CONFLICT_FREE_OFFSET(r);
                    int t = temp[l];
                    temp[l] = temp[r];
                    temp[r] += t;
                }
            }
            __syncthreads();

            if (blockOffset + ai < n)
            {
                odata[blockOffset + ai] = temp[ai + bankOffsetA];
            }
            if (blockOffset + bi < n)
            {
                odata[blockOffset + bi] = temp[bi + bankOffsetB];
            }
        }

        __global__ void kernAddBlockOffsets(int n, int *data,
                const int *blockOffsets)
        {
            int blockOffset = blockIdx.x * elemsPerBlock;
            int add = blockOffsets[blockIdx.x];

            int ai = blockOffset + threadIdx.x;
            int bi = ai + blockDim.x;
            if (ai < n)
            {
                data[ai] += add;
            }
            if (bi < n)
            {
                data[bi] += add;
            }
        }

        // Recursively scans block sums; `sums` holds one pre-allocated buffer per level.
        void devScanShared(int n, int *dev_data, const std::vector<int*>& sums,
                int level)
        {
            int numBlocks = (n + elemsPerBlock - 1) / elemsPerBlock;
            int shmemBytes =
                (elemsPerBlock + CONFLICT_FREE_OFFSET(elemsPerBlock - 1)) * sizeof(int);

            if (numBlocks == 1)
            {
                kernScanBlockShared<<<1, BLOCK_SIZE, shmemBytes>>>(
                    n, dev_data, dev_data, nullptr);
                return;
            }

            int *blockSums = sums[level];
            kernScanBlockShared<<<numBlocks, BLOCK_SIZE, shmemBytes>>>(
                n, dev_data, dev_data, blockSums);
            devScanShared(numBlocks, blockSums, sums, level + 1);
            kernAddBlockOffsets<<<numBlocks, BLOCK_SIZE>>>(n, dev_data, blockSums);
        }

        void scan(int n, int *odata, const int *idata)
        {
            int *dev_data;
            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_data failed!");

            std::vector<int*> sums;
            for (int cur = n; ; )
            {
                int numBlocks = (cur + elemsPerBlock - 1) / elemsPerBlock;
                if (numBlocks == 1)
                {
                    break;
                }
                int *buf;
                cudaMalloc((void**)&buf, numBlocks * sizeof(int));
                checkCUDAError("cudaMalloc block sums failed!");
                sums.push_back(buf);
                cur = numBlocks;
            }

            timer().startGpuTimer();
            devScanShared(n, dev_data, sums, 0);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            for (int *buf : sums)
            {
                cudaFree(buf);
            }
            cudaFree(dev_data);
        }
    }
}
