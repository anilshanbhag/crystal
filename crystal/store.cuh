#pragma once

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStoreDirect(
    int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD]
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    thread_itr[ITEM * BLOCK_THREADS] = items[ITEM];
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStoreDirect(
    int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD],
    int num_items
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      thread_itr[ITEM * BLOCK_THREADS] = items[ITEM];
    }
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStore(
    T* out,
    T  (&items)[ITEMS_PER_THREAD],
    int num_items
    ) {
  T* block_itr = out;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockStoreDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items);
  } else {
    BlockStoreDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items, num_items);
  }
}

#if 0

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStoreDirect(
    int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD]
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStoreDirect(
    int tid,
    T* block_itr,
    T  (&items)[ITEMS_PER_THREAD]
    int num_items
    ) {
  T* thread_itr = block_itr + tid;

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
  }
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStore(
    T* inp,
    T  (&items)[ITEMS_PER_THREAD]
    int num_items
    ) {
  T* block_itr = inp + blockIdx.x * blockDim.x;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockStoreDirect(threadIdx.x, block_itr, items);
  } else {
    BlockStoreDirect(threadIdx.x, block_itr, items, num_items);
  }
}

#endif

