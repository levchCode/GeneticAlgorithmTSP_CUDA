#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#define CHECK_CU_ERROR(err, cufunc)                                     \
  if (err != CUDA_SUCCESS)                                              \
    {                                                                   \
      printf ("Error %d for CUDA Driver API function '%s'.\n",          \
              err, cufunc);                                             \
      exit(-1);                                                         \
    }

#define THREADS 50
#define N 913
#define HALF_THREADS ((THREADS / 2)+THREADS)
#define BLOCKS 10

#define RAND_MAX_GA 3571

#define specimenbits 4
typedef struct specimen {
	float fitness;
	unsigned int p[specimenbits];
	unsigned int q[specimenbits];
} specimen;

#define pmutation 0.08
#define pcross 0.30

int genetic_algorithm();

#endif //GENETIC_ALGORITHM_H