#pragma once

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(
    T  item,
    T* shared
    ) {
  __syncthreads();

  T val = item;
  const int warp_size = 32;
  int lane = threadIdx.x % warp_size;
  int wid = threadIdx.x / warp_size;

  // Calculate sum across warp
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  // Store sum in buffer
  if (lane == 0) {
    shared[wid] = val;
  }

  __syncthreads();

  // Load the sums into the first warp
  val = (threadIdx.x < blockDim.x / warp_size) ? shared[lane] : 0;

  // Calculate sum of sums
  if (wid == 0) {
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
  }

  return val;
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(
    T (&items)[ITEMS_PER_THREAD],
    T* shared
    ) {
  T thread_sum = 0;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    thread_sum += items[ITEM];
  }

  return BlockSum(thread_sum, shared);
}
