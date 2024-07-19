CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) PrimedErwin
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 11, i9-13950HX @ 2.20GHz 64GB, RTX 3500 12282MB (CWC 257 Lab)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

## Implement
### CPU scan & stream compaction
a simple loop

### GPU naive scan
This is implemented with the reference to [3-Parallel-Algorithms-1.pptx - Google 幻灯片](https://docs.google.com/presentation/d/1ETVONA7QDM-WqsEj4qVOGD6Kura5I6E9yqH-7krnwZ0/edit?pli=1#slide=id.p31), page 20.
Here are a few differences between reduction and scan. For reduction, simply calculate the sum per block, then add them up to get the final sum, you don't need to consider sync. But in scan (no matter inclusive or exclusive), the current step relies on the former one. So between every op, a sync is needed. For a single block, \_\_syncthreads() is enough. For the same reason, additional buffer is needed. 
Naive scan is slower than CPU.
Tip: with RTX3500Ada, capability 8.9, hardware accelaration can be implemented with cg::exclusive_scan(), then combine each group of results together. 

### GPU efficient scan and compaction
In this section, the thought of binary tree is implemented. Binary tree has 2 phases, up-sweep and down-sweep, after 2 phases, the array's value is replaced with prefix num. Then, with map and scatter operations, the compaction can be finished.
Among these operations, scan is the most important part. Considering a very long array, we need more blocks to handle it. Though we don't need an additional buffer due to the benefit of binary tree, a sync is still needed between two operations. So how to sync all the threads in different blocks is the problem. 

One of the solutions is cooperative groups. grid_group can sync all the threads in current grid by this_grid().sync() (or something else). But this need compilation option -rdc=true(I did this), compute capability above 6.0(I did this too), cudaLaunchCooperativeKernel. But my program always crashes at cudaLaunchCooperativeKernel. I don't know why, no dynamic parallelism in my kernel.

Another solution is implicit synchronization. Take a look at up-sweep and down-sweep algorithms, their for loop can be split up to simple kernels, then with a for loop, with each time a kernel runs with multiple blocks, all the threads will sync implicitly.

### GPU thrust scan and compaction
It's mainly implemented by thrust::exclusive_scan and thrust::remove_if.


### Radix Sort
Radix sort doesn't compact stream. It uses exclusive scan to sort. It's useful to short bit arrays. Radix sort is used in one thread block in a SM, so the results are sorted chunks. Then use recurisve merge to combine two sorted chunks into one. 