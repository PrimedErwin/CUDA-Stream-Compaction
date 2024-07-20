#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include "hac.h"

namespace cg = cooperative_groups;


namespace StreamCompaction {
	namespace HAC {
		constexpr int block_size = 512;

		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		__device__
			int group_need_work(int* work_group_index, int size, int groupID)
		{
			for (int i = 0; i < size; i++)
			{
				if (groupID == work_group_index[i])
				{
					return 1;
				}
			}
			return 0;
		}

		__global__
			void tiled_scan(int n, int* odata, int* idata, int level)
		{
			extern __shared__ int work_group_index[];
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= n) return;
			cg::thread_block cta = cg::this_thread_block();
			auto tile32 = cg::tiled_partition<32>(cta);
			int group_rank = tile32.meta_group_rank();

			if (blockIdx.x == 0)
			{
				if (group_rank == 0)
				{
					//odata[idx] = cg::exclusive_scan(tile32, idata[idx]);
					idata[idx] = cg::exclusive_scan(tile32, idata[idx]);
				}
				else
				{
					//odata[idx] = cg::inclusive_scan(tile32, idata[idx]);
					idata[idx] = cg::inclusive_scan(tile32, idata[idx]);

				}
				cta.sync();
			}
			else
			{
				idata[idx] = cg::inclusive_scan(tile32, idata[idx]);

			}

			//okay let's upsweep
			//we need half of the threads each time, assume here are 8 groups
			//1st round: 1,3,5,7 operates 0,2,4,6
			//2nd round: 2,3,6,7 operates 0,1,4,5
			//3rd round: 4,5,6,7 operates 0,1,2,3
			for (int d = 0; d < level; d++)
			{
				//if (group_rank % (2<<d) == 1)
				//{
				//	idata[idx] += idata[group_rank * 32 - 1];
				//}

				//1st round it's 1 group, 2nd is 2 groups work together
				int work_group_size = (1 << d);
				//int work_group_num = block_size / 32 / 2;
				int temp_index = 0;
				for (int i = block_size / 32 - 1; i >= 0; i -= work_group_size)
				{
					for (int j = work_group_size; j > 0; j--)
					{
						work_group_index[temp_index++] = i--;
						//if (idx == 0)printf("%d round, %d index, %d\n", d, temp_index - 1, i + 1);
					}
				}
				//now all the groups that need to work has been stored in array
				if (group_need_work(work_group_index, temp_index, group_rank))
				{
					//if(tile32.thread_rank()==0) printf("%d Group %d in \n", d, group_rank);
					//minus work_gourp_size to get the group rank that needs operate
					idata[idx] += idata[(group_rank-work_group_size+1) * 32 - 1];
				}
				cta.sync();
			}
		}

		//we have a lot of blocks with each one full of prefix sums
		__global__
			void block_scan()
		{

		}






		void scan(int n, int* odata, const int* idata)
		{
			int level = ilog2ceil(n);//incomplete binary tree
			int origin_n = n;
			n = 1 << level;
			int level_32 = ilog2ceil(block_size / 32);

			dim3 blockSize(block_size);
			dim3 gridSize((n - 1) / blockSize.x + 1);

			int level_block_512 = ilog2ceil(gridSize.x);
			int* g_odata, * g_idata;
			cudaMalloc(&g_odata, n * sizeof(int));
			cudaMalloc(&g_idata, n * sizeof(int));
			cudaMemset(g_odata, 0, n * sizeof(int));
			cudaMemset(g_idata, 0, n * sizeof(int));

			cudaMemcpy(g_idata, idata, origin_n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startCpuTimer();
			tiled_scan << <gridSize, blockSize, (block_size / 64) * sizeof(int), 0 >> > (n, g_odata, g_idata, level_32);
			checkCUDAError("tiled_scan");
			//above we get prefix sums by block
			//the following performs a block scan

			timer().endCpuTimer();

			cudaMemcpy(odata, g_idata, origin_n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("hac_memcpy");
			cudaFree(g_idata);
			cudaFree(g_odata);

		}

		int compact(int n, int* odata, const int* idata)
		{

			return -1;
		}


	}
}