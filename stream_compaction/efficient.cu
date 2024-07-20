#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include "common.h"
#include "efficient.h"

namespace cg = cooperative_groups;

namespace StreamCompaction {
	namespace Efficient {
		constexpr auto block_size = 512;
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__global__
			void upsweep(int n, int* data, int d, int offset, int last_offset)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= n) return;
			if (idx % (2 << d) == 0)
			{
				data[idx + offset] += data[idx + last_offset];
			}
		}
		__global__
			void downsweep(int n, int* data, int d, int offset, int last_offset)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= n) return;
			int temp_down;
			if (idx % (2 << d) == 0)
			{
				temp_down = data[idx + last_offset];
				data[idx + last_offset] = data[idx + offset];
				data[idx + offset] += temp_down;
			}
		}

		void sweep_scan(int n, int* idata, int level, dim3 gridSize, dim3 blockSize)
		{
			int offset = 1;
			int last_offset = 0;
			int zero = 0;
			for (int d = 0; d < level; d++)
			{
				upsweep<<<gridSize, blockSize>>>(n, idata, d, offset, last_offset);
				last_offset = offset;
				offset += (2 << d);
				checkCUDAError("up sweep scan");

			}
			offset = last_offset;
			last_offset -= (1 << (level - 1));
			cudaMemcpy(&idata[n-1], const_cast<const int*>(&zero), sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("memcpy");

			for (int d = level - 1; d >= 0; d--)
			{
				downsweep<<<gridSize, blockSize>>>(n, idata, d, offset, last_offset);
				offset = last_offset;
				last_offset -= (d == 0 ? 0 : (1 << (d - 1)));
				checkCUDAError("down sweep scan");

			}
		}

		__global__//DEPRECATED
			void sweep_scan(int n, int* odata, int* idata, int level)
		{
			//cg::grid_group cta = cg::this_grid();
			//printf("n=%d\n", cta.size());
			extern __shared__ int temp[];
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= n)return;
			int offset = 1;
			int last_offset = 0;
			int temp_down;
			temp[idx] = idata[idx];
			//UP-SWEEP
			//considering power-of-two first
			//1st round: 0,2,4,6 + 1 operates 1,3,5,7+0,2,4,6
			//2nd round: 0,4 + 3 operates 3,7+1,5
			//3rd round: 0 + 7 operates 7+3
			for (int d = 0; d < level; d++)
			{
				if (idx % (2 << d) == 0)
				{
					temp[idx + offset] += temp[idx + last_offset];
				}
				last_offset = offset;
				offset += (2 << d);
				__syncthreads();
				// you can't use threadfence to sync here
				//__threadfence()
				// cg::sync() needs LaunchCooperativeKernel
				//cta.sync();
			}
			//DOWN-SWEEP

			offset = last_offset;
			last_offset -= (1 << (level - 1));
			//now offset is 7, last_offset is 3, still consider power-of-two
			//
			if (idx == n - 1) temp[idx] = 0;
			for (int d = level - 1; d >= 0; d--)
			{
				if (idx % (2 << d) == 0)
				{
					temp_down = temp[idx + last_offset];
					temp[idx + last_offset] = temp[idx + offset];
					temp[idx + offset] += temp_down;
				}
				offset = last_offset;
				last_offset -= (d == 0 ? 0 : (1 << (d - 1)));
				__syncthreads();
				//__threadfence();
				//cta.sync();
			}
			odata[idx] = temp[idx];
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {
			//size above 1024 need more blocks, but how to sync all the threads in a grid?
			//solution1 is to seperate the scan func, use for loop to do each op(implicit sync)
			//solution2 use cooperative groups. I give up, kernel launch always fails.
			int level = ilog2ceil(n);//incomplete binary tree
			//Here consider non-power-of-two, we need to pad 0s behind
			//So first round up n to nearest power-of-two
			int origin_n = n;
			n = 1 << level;

			dim3 blockSize(block_size);
			dim3 gridSize((n - 1) / blockSize.x + 1);

			int* g_odata;
			cudaMalloc(&g_odata, n * sizeof(int));
			cudaMemset(g_odata, 0, n * sizeof(int));
			//cudaMemset(g_idata, 0, n * sizeof(int));
			//copy mem to device
			cudaMemcpy(g_odata, idata, origin_n * sizeof(int), cudaMemcpyHostToDevice);
			timer().startGpuTimer();
			// TODO
			//this O(n) algorithm contains 2 phase
			//Sweep
			//sweep_scan << <gridSize, blockSize, n * sizeof(int), 0 >> > (n, g_odata, g_odata, level);
			sweep_scan(n, g_odata, level, gridSize, blockSize);
			checkCUDAError("sweep scan");
			//void* kernelArgs[] = { &n, g_odata, g_idata, &level };
			//cudaLaunchCooperativeKernel((void *)sweep_scan, gridSize, blockSize, kernelArgs, n * sizeof(int), 0);
			//checkCUDAError("Error launching cooperative kernel");
			timer().endGpuTimer();
			//copy mem to host
			cudaMemcpy(odata, g_odata, origin_n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(g_odata);
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
		int compact(int n, int* odata, const int* idata) {
			int level = ilog2ceil(n);
			int origin_n = n;
			int compacted_num;
			n = 1 << level;
			dim3 blockSize(block_size);
			dim3 gridSize((n - 1) / blockSize.x + 1);
			int* g_odata, * g_idata, * g_bools, *g_indice;
			cudaMalloc(&g_odata, n * sizeof(int));
			cudaMalloc(&g_idata, n * sizeof(int));
			cudaMalloc(&g_bools, n * sizeof(int));
			cudaMalloc(&g_indice, n * sizeof(int));
			cudaMemset(g_idata, 0, n * sizeof(int));
			cudaMemset(g_bools, 0, n * sizeof(int));
			cudaMemset(g_odata, 0, n * sizeof(int));
			cudaMemset(g_indice, 0, n * sizeof(int));
			//copy mem to device
			cudaMemcpy(g_idata, idata, origin_n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();
			// TODO
			StreamCompaction::Common::kernMapToBoolean << <gridSize, blockSize >> > (n, g_bools, g_idata);
			//sweep_scan << <gridSize, blockSize, n * sizeof(int), 0 >> > (n, g_odata, g_bools, level);
			cudaMemcpy(g_indice, g_bools, origin_n * sizeof(int), cudaMemcpyDeviceToDevice);
			sweep_scan(n, g_indice, level, gridSize, blockSize);
			StreamCompaction::Common::kernScatter << <gridSize, blockSize >> > (n, g_odata, g_idata, g_bools, g_indice);
			timer().endGpuTimer();
			//cudaDeviceSynchronize();

			if (origin_n == n) cudaMemcpy(&compacted_num, &g_indice[origin_n - 1], sizeof(int), cudaMemcpyDeviceToHost);
			else cudaMemcpy(&compacted_num, &g_indice[origin_n], sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(odata, g_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(g_odata);
			cudaFree(g_idata);
			cudaFree(g_bools);
			cudaFree(g_indice);

			return compacted_num;
		}
	}
}
