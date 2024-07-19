#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
	namespace Naive {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		// TODO: __global__
		//exclusive scan implemented on gpu
		__global__
			void naive_scan(int n, int* odata, int* idata, int* temp, int logceil)
		{
			unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			int* p1 = idata;
			int* p2 = temp;
			int* ptemp;
			temp[idx] = (idx>0) ? idata[idx - 1] : 0;
			__syncthreads();
			for (int d = 0; d < logceil; d++)
			{
				if (idx >= (1 << d))
				{
					p2[idx] = p1[idx] + p1[idx - (1 << d)];
				}
				else
				{
					p2[idx] = p1[idx];
				}
				ptemp = p1;
				p1 = p2;
				p2 = ptemp;
				__syncthreads();
			}
			odata[idx+1] = (logceil % 2) ? (p2[idx]) : (p1[idx]);
		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {
			// TODO
			dim3 gridSize(1);
			dim3 blockSize(n);
			//alloc mem, since all threads is not guaranteed to work at the same time
			//additional temp buffer is needed
			int* g_idata, * g_odata, * temp;
			cudaMalloc(&g_idata, n * sizeof(int));
			cudaMalloc(&temp, n * sizeof(int));
			cudaMalloc(&g_odata, n * sizeof(int));
			//copy mem
			cudaMemcpy(g_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			//record time of calculation
			timer().startGpuTimer();

			naive_scan << <gridSize, blockSize >> > (n, g_odata, g_idata, temp, ilog2ceil(n));

			timer().endGpuTimer();
			//copy mem
			cudaMemcpy(odata, g_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			//free mem
			cudaFree(g_idata);
			cudaFree(g_odata);
			cudaFree(temp);

		}
	}
}
