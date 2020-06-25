// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <cmath>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include "cub/test/test_util.h"

#include "crystal/crystal.cuh"

#include "utils/gpu_utils.h"

using namespace std;


//---------------------------------------------------------------------
// Implements Projection Operator
// There are two variants: dot-product and sigmoid
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void project(float* in1, float* in2, float* out, int num_items)
{
  float items[ITEMS_PER_THREAD];
  float items2[ITEMS_PER_THREAD];
  float res[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_items + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_items - tile_offset;
  }

  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in1 + tile_offset, items, num_tile_items);
  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in2 + tile_offset, items2, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (threadIdx.x + (ITEM * BLOCK_THREADS) < num_tile_items) {
      res[ITEM] = 2*items[ITEM] + 3*items2[ITEM];
    }
  }

  BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>(out + tile_offset, res, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void projectSigmoid(float* in1, float* in2, float* out, int num_items)
{
  float items[ITEMS_PER_THREAD];
  float items2[ITEMS_PER_THREAD];
  float res[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_items + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_items - tile_offset;
  }

  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in1 + tile_offset, items, num_tile_items);
  BlockLoad<float, BLOCK_THREADS, ITEMS_PER_THREAD>(in2 + tile_offset, items2, num_tile_items);

  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (threadIdx.x + (ITEM * BLOCK_THREADS) < num_tile_items) {
      res[ITEM] = 1.0f / (1.0f + expf(-2*items[ITEM] -3*items2[ITEM]));
    }
  }

  BlockStore<float, BLOCK_THREADS, ITEMS_PER_THREAD>(out + tile_offset, res, num_tile_items);
}


float projectGPU(float* in1, float* in2, float* out, int num_items) {
  SETUP_TIMING();

  float time_proj;
  int tile_items = 128*4;
  int num_blocks = (num_items + tile_items - 1)/tile_items;
  TIME_FUNC((project<128,4><<<num_blocks, 128>>>(in1, in2, out, num_items)), time_proj);

  return time_proj;
}

float projectSigmoidGPU(float* in1, float* in2, float* out, int num_items) {
  SETUP_TIMING();

  float time_proj;
  int tile_items = 128*4;
  int num_blocks = (num_items + tile_items - 1)/tile_items;
  TIME_FUNC((projectSigmoid<128,4><<<num_blocks, 128>>>(in1, in2, out, num_items)), time_proj);

  return time_proj;
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_items           = 1<<28;
  int num_trials          = 1;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("n", num_items);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
      printf("%s "
          "[--n=<input items>] "
          "[--t=<num trials>] "
          "[--device=<device-id>] "
          "[--v] "
          "\n", argv[0]);
      exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Allocate problem device arrays
  float *d_in1 = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in1, sizeof(float) * num_items));

  float *d_in2 = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in2, sizeof(float) * num_items));

  float  *d_out = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * num_items));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  curandGenerator_t generator;
  int seed = 0;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator,seed);
  curandGenerateUniform(generator, d_in1, num_items);
  curandGenerateUniform(generator, d_in2, num_items);

  float time_proj_gpu;
  float time_proj_sigmoid_gpu;  

  for (int t = 0; t < num_trials; t++) {
    time_proj_gpu = projectGPU(d_in1, d_in2, d_out, num_items);
    time_proj_sigmoid_gpu = projectSigmoidGPU(d_in1, d_in2, d_out, num_items);

    cout<< "{"
        << "\"time_proj_gpu\":" << time_proj_gpu
        << ",\"time_proj_sigmoid_gpu\":" << time_proj_sigmoid_gpu
        << "}" << endl;
  }

  // Cleanup
  if (d_in1) CubDebugExit(g_allocator.DeviceFree(d_in1));
  if (d_in2) CubDebugExit(g_allocator.DeviceFree(d_in2));
  if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));

  return 0;
}

