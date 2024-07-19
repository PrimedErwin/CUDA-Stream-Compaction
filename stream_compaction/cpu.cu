#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
	namespace CPU {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		/**
		 * CPU scan (prefix sum).
		 * For performance analysis, this is supposed to be a simple for loop.
		 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
		 */
		void scan(int n, int* odata, const int* idata) {
			//timer().startCpuTimer();
			// TODO
			//odata[0] = idata[0];//exclusive scan array[0] is 0;
			odata[0] = 0;
			for (int i = 1; i < n; i++)
			{
				odata[i] = idata[i - 1] + odata[i - 1];
			}
			//timer().endCpuTimer();
		}

		/**
		 * CPU stream compaction without using the scan function.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithoutScan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			// TODO
			int num_remain = 0;
			for (int i = 0; i < n; i++)
			{
				if (idata[i]) odata[num_remain++] = idata[i];
			}
			timer().endCpuTimer();
			return num_remain;
		}

		/**
		 * CPU stream compaction using scan and scatter, like the parallel version.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithScan(int n, int* odata, const int* idata) {
			timer().startCpuTimer();
			// TODO
			int* criterion = (int*)malloc(n * sizeof(int));
			//int criterion[(1 << 4)];
			if (criterion == NULL) exit(EXIT_FAILURE);
			//find number > 0, and replace criterion[i] with 1
			for (int i = 0; i < n; i++)
			{
				criterion[i] = (idata[i] ? 1 : 0);
			}
			//compute exclusive scan of criterion
			//index 0 of exclusive scan is 0, so the value of exclusive scan
			//is the index of odata
			StreamCompaction::CPU::scan(n, odata, criterion);
			for (int i = 0; i < n; i++)
			{
				if (criterion[i]) odata[odata[i]] = idata[i];
			}
			timer().endCpuTimer();
			return odata[n-1];
		}
	}
}
