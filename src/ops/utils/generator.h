#pragma once

#include <iostream>
#include <cstdio>              /* perror */
#include <cstdlib>             /* posix_memalign */
#include <immintrin.h>
#include <thread>
using namespace std;

#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))
#define RANDR_RANGE(N) ((double)rand_r(&seed) / ((double)RAND_MAX + 1) * (N))
static int seeded = 0;

/** Check wheter seeded, if not seed the generator with current time */
static void
check_seed()
{
    if(!seeded) {
        srand(0);
        seeded = 1;
    }
}

/**
 * Shuffle tuples of the relation using Knuth shuffle.
 *
 * @param relation
 */
void
knuth_shuffle(int* arr, int num_tuples)
{
    int i;
    for (i = num_tuples - 1; i > 0; i--) {
        int  j              = RAND_RANGE(i);
        int tmp             = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}


/**
 * Generate unique tuple IDs with Knuth shuffling
 * relation must have been allocated
 */
void
random_unique_gen(int*& arr, int num_tuples)
{
  int i;

  for (i = 0; i < num_tuples; i++) {
    arr[i] = (i+1);
  }

  /* randomly shuffle elements */
  knuth_shuffle(arr, num_tuples);
}

void
dummy_initialize(int*& arr, int num_tuples) {
    for (int i = 0; i < num_tuples; i++) {
        arr[i] = i;
    }
}

int
create_relation_pk(int*& keys, int*& vals, int num_tuples)
{
  check_seed();

  keys = (int*)_mm_malloc(num_tuples * sizeof(int), 256);
  vals = (int*)_mm_malloc(num_tuples * sizeof(int), 256);

  if (!keys || !vals) {
      perror("out of memory");
      return -1;
  }

  random_unique_gen(keys, num_tuples);
  dummy_initialize(vals, num_tuples);

  return 0;
}

int create_relation_fk(int*& keys, int*& vals, int num_tuples, const int maxid)
{
  int i, iters, remainder;

  check_seed();
  keys = (int*)_mm_malloc(num_tuples * sizeof(int), 256);
  vals = (int*)_mm_malloc(num_tuples * sizeof(int), 256);

  if (!keys || !vals) {
    perror("out of memory");
    return -1;
  }

  // alternative generation method
  iters = num_tuples / maxid;
  for (i = 0; i < iters; i++) {
    int* tuples = keys + maxid * i;
    random_unique_gen(tuples, maxid);
  }

  // if num_tuples is not an exact multiple of maxid
  remainder = num_tuples % maxid;
  if (remainder > 0) {
    int* tuples = keys + maxid * iters;
    random_unique_gen(tuples, remainder);
  }

  dummy_initialize(vals, num_tuples);
  return 0;
}

/*
typedef struct rand_state_64 {
  uint64_t num[313];
  size_t index;
} rand64_t;

rand64_t *rand64_init(uint64_t seed)
{
  rand64_t *state = malloc(sizeof(rand64_t));
  uint64_t *n = state->num;
  size_t i;
  n[0] = seed;
  for (i = 0 ; i != 311 ; ++i)
    n[i + 1] = 6364136223846793005ull *
               (n[i]  (n[i] >> 62)) + i + 1;
  state->index = 312;
  return state;
}

uint64_t rand64_next(rand64_t *state)
{
  uint64_t x, *n = state->num;
  if (state->index == 312) {
    size_t i = 0;
    do {
      x = n[i] & 0xffffffff80000000ull;
      x |= n[i + 1] & 0x7fffffffull;
      n[i] = n[i + 156]  (x >> 1);
      n[i] = 0xb5026f5aa96619e9ull & -(x & 1);
    } while (++i != 156);
    n[312] = n[0];
    do {
      x = n[i] & 0xffffffff80000000ull;
      x |= n[i + 1] & 0x7fffffffull;
      n[i] = n[i - 156]  (x >> 1);
      n[i] = 0xb5026f5aa96619e9ull & -(x & 1);
    } while (++i != 312);
    state->index = 0;
  }
  x = n[state->index++];
  x = (x >> 29) & 0x5555555555555555ull;
  x = (x << 17) & 0x71d67fffeda60000ull;
  x = (x << 37) & 0xfff7eee000000000ull;
  x = (x >> 43);
  return x;
}

typedef struct rand_state_32 {
  uint32_t num[625];
  size_t index;
} rand32_t;

rand32_t *rand32_init(uint32_t seed)
{
  rand32_t *state = malloc(sizeof(rand32_t));
  uint32_t *n = state->num;
  size_t i;
  n[0] = seed;
  for (i = 0 ; i != 623 ; ++i)
    n[i + 1] = 0x6c078965 * (n[i]  (n[i] >> 30));
  state->index = 624;
  return state;
}

uint32_t rand32_next(rand32_t *state)
{
  uint32_t y, *n = state->num;
  if (state->index == 624) {
    size_t i = 0;
    do {
      y = n[i] & 0x80000000;
      y += n[i + 1] & 0x7fffffff;
      n[i] = n[i + 397]  (y >> 1);
      n[i] = 0x9908b0df & -(y & 1);
    } while (++i != 227);
    n[624] = n[0];
    do {
      y = n[i] & 0x80000000;
      y += n[i + 1] & 0x7fffffff;
      n[i] = n[i - 227]  (y >> 1);
      n[i] = 0x9908b0df & -(y & 1);
    } while (++i != 624);
    state->index = 0;
  }
  y = n[state->index++];
  y = (y >> 11);
  y = (y << 7) & 0x9d2c5680;
  y = (y << 15) & 0xefc60000;
  y = (y >> 18);
  return y;
}

static int hardware_threads(void)
{
  char name[64];
  struct stat st;
  int threads = -1;
  do {
    sprintf(name, "/sys/devices/system/cpu/cpu%d", ++threads);
  } while (stat(name, &st) == 0);
  return threads;
}

static void *mamalloc(size_t size)
{
  void *p = NULL;
  return posix_memalign(&p, 64, size) ? NULL : p;
}

typedef struct {
  pthread_t id;
  int seed;
  int thread;
  int threads;
  uint32_t hash_factor;
  uint32_t invalid_key;
  uint32_t *inner;
  uint32_t *outer;
  volatile uint32_t *table;
  size_t inner_size;
  size_t outer_size;
  size_t table_size;
  size_t join_size;
  double selectivity;
  pthread_barrier_t *barrier;
} info_t;

static void *run(void *arg)
{
  info_t *d = (info_t*) arg;
  assert(pthread_equal(pthread_self(), d->id));
  int thread = d->thread;
  int threads = d->threads;
  uint32_t hash_factor = d->hash_factor;
  uint32_t invalid_key = d->invalid_key;
  uint32_t *inner = d->inner;
  uint32_t *outer = d->outer;
  volatile uint32_t *table = d->table;
  size_t i, o, t, h;
  size_t inner_size = d->inner_size;
  size_t outer_size = d->outer_size;
  size_t table_size = d->table_size;
  size_t inner_beg = (inner_size / threads) *  thread;
  size_t inner_end = (inner_size / threads) * (thread + 1);
  size_t outer_beg = (outer_size / threads) *  thread;
  size_t outer_end = (outer_size / threads) * (thread + 1);
  size_t table_beg = (table_size / threads) *  thread;
  size_t table_end = (table_size / threads) * (thread + 1);
  if (thread + 1 == threads) {
    inner_end = inner_size;
    outer_end = outer_size;
    table_end = table_size;
  }
  for (t = table_beg ; t != table_end ; ++t)
    table[t] = invalid_key;
  pthread_barrier_wait(&d->barrier[0]);
  rand32_t *gen = rand32_init(d->seed);
  for (i = inner_beg ; i != inner_end ; ++i) {
    int new_key_inserted = 0;
    uint32_t key;
    do {
      do {
        key = rand32_next(gen);
      } while (key == invalid_key);
      h = (uint32_t) (key * hash_factor);
      h = (h * table_size) >> 32;
      for (;;) {
        if (table[h] == invalid_key &&
            __sync_bool_compare_and_swap(&table[h], invalid_key, key)) {
            new_key_inserted = 1;
          break;
        }
        if (table[h] == key) break;
        if (++h == table_size) h = 0;
      }
    } while (new_key_inserted == 0);
    inner[i] = key;
  }
  pthread_barrier_wait(&d->barrier[1]);
  size_t join_size = 0;
  uint32_t limit = ~0;
  limit *= d->selectivity;
  for (o = outer_beg ; o != outer_end ; ++o) {
    uint32_t key;
    if (rand32_next(gen) <= limit) {
      i = rand32_next(gen);
      i = (i * inner_size) >> 32;
      key = inner[i];
      join_size++;
    } else do {
      do {
        key = rand32_next(gen);
      } while (key == invalid_key);
      h = (uint32_t) (key * hash_factor);
      h = (h * table_size) >> 32;
      while (table[h] != invalid_key) {
        if (table[h] == key) break;
        if (++h == table_size) h = 0;
      }
    } while (table[h] == key);
    outer[o] = key;
  }
  free(gen);
  d->join_size = join_size;
  pthread_exit(NULL);
}

size_t inner_outer(size_t inner_size, size_t outer_size, double selectivity,
                   uint32_t **inner_p, uint32_t **outer_p)
{
  srand(time(NULL));
  int t, threads = hardware_threads();
  // input arguments
  assert(inner_size <= 1000 * 1000 * 1000);
  assert(selectivity >= 0.0 && selectivity <= 1.0);
  // tables
  uint32_t *inner = mamalloc((inner_size + 1) * sizeof(uint32_t));
  uint32_t *outer = mamalloc(outer_size * sizeof(uint32_t));
  size_t table_size = inner_size / 0.7;
  uint32_t *table = malloc(table_size * sizeof(uint32_t));
  // constants
  uint32_t hash_factor = (rand() << 1) | 1;
  uint32_t invalid_key = rand() * rand();
  // barriers
  int b, barriers = 2;
  pthread_barrier_t barrier[barriers];
  for (b = 0 ; b != barriers ; ++b)
    pthread_barrier_init(&barrier[b], NULL, threads);
  // run threads
  info_t info[threads];
  for (t = 0 ; t != threads ; ++t) {
    info[t].seed = rand();
    info[t].thread = t;
    info[t].threads = threads;
    info[t].hash_factor = hash_factor;
    info[t].invalid_key = invalid_key;
    info[t].selectivity = selectivity;
    info[t].inner = inner;
    info[t].outer = outer;
    info[t].table = table;
    info[t].inner_size = inner_size;
    info[t].outer_size = outer_size;
    info[t].table_size = table_size;
    info[t].barrier = barrier;
    pthread_create(&info[t].id, NULL, run, (void*) &info[t]);
  }
  size_t join_size = 0;
  for (t = 0 ; t != threads ; ++t) {
    pthread_join(info[t].id, NULL);
    join_size += info[t].join_size;
  }
  // cleanup
  for (b = 0 ; b != barriers ; ++b)
    pthread_barrier_destroy(&barrier[b]);
  free(table);
  // pass output
  inner[inner_size] = invalid_key;
  *inner_p = inner;
  *outer_p = outer;
  return join_size;
}
*/
