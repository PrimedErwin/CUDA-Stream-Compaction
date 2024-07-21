#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, bool b);

        int compactWithoutScan(int n, int *odata, const int *idata);

        int compactWithScan(int n, int *odata, const int *idata);
    }
}
