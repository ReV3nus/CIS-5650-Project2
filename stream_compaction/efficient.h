#pragma once

#include "common.h"

namespace StreamCompaction
{
    namespace Efficient
    {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scanUnoptimized(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        // In-place exclusive scan on a power-of-two, zero-padded device buffer.
        // Exposed for reuse by other modules (e.g. radix sort).
        void devScan(int paddedN, int *dev_data);
    }
}
