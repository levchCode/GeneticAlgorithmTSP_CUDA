#include <cuda.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <time.h>

#include <stdio.h>

#include "genetic_algorithm.h"

template <class T> void swap (T& a, T& b){
	T c(a); a = b; b = c;
}

__device__ int rand_int(int *seed){
	unsigned int xi = *(unsigned int *)seed;
	unsigned int m = 65537 * 67777;

	xi = (xi * xi) % m;
	*seed = *(unsigned int *)&xi;

	return xi % RAND_MAX_GA;
}

__device__ float rand_float(int *seed){
	float r = (float)(rand_int(seed) % 100);
	return r / 100.0;
}

__device__ int selectSpecimen(specimen *pop, int size, int *random_seed){
	int i, j;
	i = rand_int(random_seed) % size;
	j = (rand_int(random_seed) % (size - 1) + i + 1) % size;

	return (pop[i].fitness < pop[j].fitness) ? i : j;
}

__device__ void crossover(specimen *parent, specimen *offspring, int *random_seed){
	int i;
	int cpoint = rand_int(random_seed) % specimenbits;
	for(i = 0; i < specimenbits; ++i){
		int part = (i < cpoint) ? 1 : 0;
		offspring[0].p[i] = parent[part].p[i];
		offspring[1].p[i] = parent[1-part].p[i];
		offspring[0].q[i] = parent[part].q[i];
		offspring[1].q[i] = parent[1 - part].q[i];
	}

	offspring[0].fitness = 0;
	offspring[1].fitness = 0;
}

__device__ void mutate(specimen *parent, int *random_seed){
	int i;

	int mp = rand_int(random_seed) % 100 - 50;
	int mq = rand_int(random_seed) % 100 - 50;

	for(i = 0; i < specimenbits; ++i){
		if(rand_float(random_seed) < pmutation){
			parent->p[i] = mp + parent->p[i];
			parent->q[i] = mq + parent->q[i];
		}
	}
}

__device__ __host__ int sum(const unsigned int *m)
	{
		int s = 0;
		int i = 0;
		for (i = 0; i < specimenbits; ++i) {
			s += (int)(m[i]);
		}
		return s;
	}

__device__ float fitness(const specimen *sp){

	return abs(N - sum(sp->p)*sum(sp->q));
}

__global__ void initPopulation(specimen *pop, const int size, const int random_seed){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < size){
		int seed = random_seed + i, j;
		for(j = 0; j < specimenbits; ++j)
			pop[i].p[j] = rand_int(&seed) % 10;
			pop[i].q[j] = rand_int(&seed) % 10;
	}
}

__global__ void newGeneration(specimen *pop, specimen *newpop, const int size, const int random_seed){
	const int i = 2 * (blockIdx.x*blockDim.x + threadIdx.x);
	if((i + 1) >= size) return;

	specimen parent[2], offspring[2];
	int seed = random_seed + i;

	parent[0] = pop[selectSpecimen(pop, size, &seed)];
	parent[1] = pop[selectSpecimen(pop, size, &seed)];

	if(rand_float(&seed) < pcross){
		crossover(parent, offspring, &seed);
	} else {
		offspring[0] = parent[0];
		offspring[1] = parent[1];
	}

	mutate(&offspring[0], &seed);
	mutate(&offspring[1], &seed);
	newpop[i] = offspring[0];
	newpop[i+1] = offspring[1];
}

__global__ void countFitness(specimen *pop, const int size){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < size){
		const specimen sp = pop[i];
		pop[i].fitness = fitness(&sp);
	}
}

__global__ void findBestSpecimen(specimen *pop, const int size){
	const int index = threadIdx.x;
	if(index >= THREADS) return;

	int bestIndex = index, i;
	for(i = index+THREADS; i < size; i += THREADS){
		if(pop[bestIndex].fitness < pop[i].fitness)
			bestIndex = i;
	}

	__shared__ int buffer[THREADS];
	buffer[index] = bestIndex;
	__syncthreads();

	if(index == 0){
		for(i = 0; i < THREADS; ++i)
			if(pop[bestIndex].fitness < pop[ buffer[i] ].fitness)
				bestIndex = buffer[i];

		pop[0] = pop[bestIndex];
	}
}



int main(){
	srand (time(NULL));

	const int population = THREADS * BLOCKS;
	specimen best;

	specimen *devPopulation = 0, *devNewPopulation = 0;
	cudaMalloc((void**)&devPopulation, sizeof(specimen) * population);
	cudaMalloc((void**)&devNewPopulation, sizeof(specimen) * population);

	initPopulation<<<BLOCKS, THREADS>>>(devPopulation, population, rand() % RAND_MAX_GA);
	cudaThreadSynchronize();

	while(true){

		findBestSpecimen << <1, THREADS >> >(devPopulation, population);
		cudaThreadSynchronize();

		cudaMemcpy(&best, &devPopulation[0], sizeof(specimen), cudaMemcpyDeviceToHost);

		int p = sum(best.p);
		int q = sum(best.q);

		if (p*q == N) {
			printf("Found: p = %d q = %d", p, q);
			break;
		}

		printf("Best fitness: %f (p = %d | q = %d)", best.fitness, p, q);
		printf("\n");

		countFitness<<<BLOCKS, THREADS>>>(devPopulation, population);
		newGeneration<<<BLOCKS, HALF_THREADS>>>(devPopulation, devNewPopulation, population, rand() % RAND_MAX_GA);
		cudaThreadSynchronize();
		swap(devPopulation, devNewPopulation); 
	}


	

	getchar();

	cudaFree(devPopulation);
	cudaFree(devNewPopulation);

	
}