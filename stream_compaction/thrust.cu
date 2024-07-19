#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int origin_n = n;
            int level = ilog2ceil(n);
            n = 1 << level;

            //create host_vectors and copy them into device
            thrust::host_vector<int> v_data_host(idata, idata+n);
            thrust::device_vector<int> v_idata(n);
            thrust::device_vector<int> v_odata(n, 0);
            v_idata = v_data_host;

            thrust::exclusive_scan(v_idata.begin(), v_idata.end(), v_odata.begin());
            
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::exclusive_scan(v_idata.begin(), v_idata.end(), v_odata.begin());
            timer().endGpuTimer();

            thrust::copy_n(v_odata.begin(), origin_n, odata);
        }

        struct is_zero{
            __host__ __device__
                bool operator()(const int x)
            {
                return x == 0;
            }
        };

        int compact(int n, int* odata, const int* idata)
        {
            int origin_n = n;
            int level = ilog2ceil(n);
            n = 1 << level;

            thrust::device_vector<int> v_idata(idata, idata+origin_n);

            int zero_count = thrust::count_if(thrust::device, v_idata.begin(), v_idata.end(), is_zero());
            //The first run of thrust costs a lot of time
            timer().startGpuTimer();
            thrust::remove_if(v_idata.begin(), v_idata.end(), is_zero());
            timer().endGpuTimer();

            thrust::copy_n(v_idata.begin(), origin_n, odata);

            return origin_n - zero_count;
        }
    }
}
