#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include "..\Feleves_MonteCarlo_Pi\PreciseTimer.cpp"

#define ITERATION 100000
#define BLOCK_SIZE 1000
#define THREADS_PER_BLOCK 256

int points_within_circle = 0;
int points_within_square = 0;

double random_x, random_y, dist, pi;

double random_x_array[ITERATION];
double random_y_array[ITERATION];
int points_within_circle_array[ITERATION];
int points_within_square_array[ITERATION];
double pi_array[ITERATION];

//double test[10];

//__device__ double dev_test[10];
__device__ int dev_points_within_circle;
__device__ int dev_points_within_square;
__device__ double dev_random_x, dev_random_y, dev_dist, dev_pi;

__device__ double dev_random_x_array[ITERATION];
__device__ double dev_random_y_array[ITERATION];
__device__ int dev_points_within_circle_array[ITERATION];
__device__ int dev_points_within_square_array[ITERATION];
__device__ double dev_pi_array[ITERATION];

static std::default_random_engine rng = std::default_random_engine{};
static std::uniform_real_distribution<float> distribution(-1.0, 1.0);

void EstimatePi() {
	for (int i = 0; i < ITERATION; i++)
	{
		random_x = distribution(rng);

		random_y = distribution(rng);

		dist = random_x * random_x + random_y * random_y;

		if (dist <= 1)
			points_within_circle++;

		points_within_square++;

		pi = double(4 * points_within_circle) / points_within_square;

		std::cout << random_x << " " << random_y << " "
			<< points_within_circle << " " << points_within_square
			<< " - " << pi << std::endl
			<< std::endl;
	}
}

__global__ void EstimatePiSingle() {
	for (int i = 0; i < ITERATION; i++)
	{
		curandState state;

		curand_init(clock64(), i, 0, &state);

		dev_random_x = curand_uniform(&state);
		dev_random_y = curand_uniform(&state);

		dev_dist = dev_random_x * dev_random_x + dev_random_y * dev_random_y;

		if (dev_dist <= 1)
			dev_points_within_circle++;

		dev_points_within_square++;

		dev_pi = double(4 * dev_points_within_circle) / dev_points_within_square;

		dev_random_x_array[i] = dev_random_x;
		dev_random_y_array[i] = dev_random_y;
		dev_points_within_circle_array[i] = dev_points_within_circle;
		dev_points_within_square_array[i] = dev_points_within_square;
		dev_pi_array[i] = dev_pi;
	}
}

__global__ void EstimatePiN() {

	int i = threadIdx.x;
	curandState state;

	curand_init(clock64(), i, 0, &state);

	dev_random_x = curand_uniform(&state);
	dev_random_y = curand_uniform(&state);

	dev_dist = dev_random_x * dev_random_x + dev_random_y * dev_random_y;

	if (dev_dist <= 1)
		dev_points_within_circle++;

	dev_points_within_square++;

	dev_pi = double(4 * dev_points_within_circle) / dev_points_within_square;

	dev_random_x_array[i] = dev_random_x;
	dev_random_y_array[i] = dev_random_y;
	dev_points_within_circle_array[i] = dev_points_within_circle;
	dev_points_within_square_array[i] = dev_points_within_square;
	dev_pi_array[i] = dev_pi;
}

__global__ void EstimatePiNWithBlocks() {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= ITERATION)
		return;

	curandState state;

	curand_init(clock64(), i, 0, &state);

	dev_random_x = curand_uniform(&state);
	dev_random_y = curand_uniform(&state);

	dev_dist = dev_random_x * dev_random_x + dev_random_y * dev_random_y;

	if (dev_dist <= 1)
		dev_points_within_circle++;  

	dev_points_within_square++;

	dev_pi = double(4 * dev_points_within_circle) / dev_points_within_square;

	dev_random_x_array[i] = dev_random_x;
	dev_random_y_array[i] = dev_random_y;
	dev_points_within_circle_array[i] = dev_points_within_circle;
	dev_points_within_square_array[i] = dev_points_within_square;
	dev_pi_array[i] = dev_pi;

}__global__ void EstimatePiShared() {
	__shared__ int shr_points_within_circle;
	__shared__ int shr_points_within_square;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	curandState state;

	curand_init(clock64(), i, 0, &state);

	dev_random_x = curand_uniform(&state);
	dev_random_y = curand_uniform(&state);

	dev_dist = dev_random_x * dev_random_x + dev_random_y * dev_random_y;

	if (dev_dist <= 1)
		dev_points_within_circle++;

	dev_points_within_square++;

	atomicAdd(&shr_points_within_circle, dev_points_within_circle);
	atomicAdd(&shr_points_within_square, dev_points_within_square);

	__syncthreads();

	if (i == 0)
	{
		atomicAdd(&dev_points_within_circle, shr_points_within_circle);
		atomicAdd(&dev_points_within_square, shr_points_within_square);

		dev_pi = double(4 * dev_points_within_circle) / dev_points_within_square;
	}

	dev_random_x_array[i] = dev_random_x;
	dev_random_y_array[i] = dev_random_y;
	dev_points_within_circle_array[i] = dev_points_within_circle;
	dev_points_within_square_array[i] = dev_points_within_square;
	dev_pi_array[i] = dev_pi;
}

__global__ void EstimatePiShared2() {
	__shared__ double shared_random_x_array[BLOCK_SIZE];
	__shared__ double shared_random_y_array[BLOCK_SIZE];

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= ITERATION)
		return;

	curandState state;

	curand_init(clock64(), i, 0, &state);

	shared_random_x_array[threadIdx.x] = curand_uniform(&state);
	shared_random_y_array[threadIdx.x] = curand_uniform(&state);

	__syncthreads();

	dev_random_x = shared_random_x_array[threadIdx.x];
	dev_random_y = shared_random_y_array[threadIdx.x];

	dev_dist = dev_random_x * dev_random_x + dev_random_y * dev_random_y;

	int dev_points_within_circle = 0;
	int dev_points_within_square = 0;

	if (dev_dist <= 1)
		dev_points_within_circle++;

	dev_points_within_square++;

	dev_pi = double(4 * dev_points_within_circle) / dev_points_within_square;

	dev_random_x_array[i] = dev_random_x;
	dev_random_y_array[i] = dev_random_y;
	dev_points_within_circle_array[i] = dev_points_within_circle;
	dev_points_within_square_array[i] = dev_points_within_square;
	dev_pi_array[i] = dev_pi;
}


void PrintOut() {
	for (int i = 0; i < ITERATION; i++)
	{
		std::cout << random_x_array[i] << " " << random_y_array[i] << " "
			<< points_within_circle_array[i] << " " << points_within_square_array[i]
			<< " - " << pi_array[i] << std::endl
			<< std::endl;
	}
}


int main()
{
	#pragma region CPU
	CPreciseTimer timer;
	timer.StartTimer();
	//EstimatePi();
	//timer.StopTimer();
	//float elapsedTime = timer.GetTimeSec();
	//std::cout << elapsedTime << std::endl;

	#pragma endregion

	#pragma region Single Thread
	cudaMemcpyToSymbol(dev_points_within_circle, &points_within_circle, sizeof(int));
	cudaMemcpyToSymbol(dev_points_within_square, &points_within_square, sizeof(int));

	/*EstimatePiSingle << < 1, 1 >> > ();
	

	cudaMemcpyFromSymbol(random_x_array, dev_random_x_array, ITERATION * sizeof(double));
	cudaMemcpyFromSymbol(random_y_array, dev_random_y_array, ITERATION * sizeof(double));
	cudaMemcpyFromSymbol(points_within_circle_array, dev_points_within_circle_array, ITERATION * sizeof(int));
	cudaMemcpyFromSymbol(points_within_square_array, dev_points_within_square_array, ITERATION * sizeof(int));
	cudaMemcpyFromSymbol(pi_array, dev_pi_array, ITERATION * sizeof(double));

	timer.StopTimer();

	PrintOut();
	float elapsedTime = timer.GetTimeSec();
	std::cout << elapsedTime << std::endl;*/

	#pragma endregion
	
	#pragma region N Threads

	//EstimatePiN << < 1, ITERATION >> > ();

	//cudaMemcpyFromSymbol(random_x_array, dev_random_x_array, ITERATION * sizeof(double));
	//cudaMemcpyFromSymbol(random_y_array, dev_random_y_array, ITERATION * sizeof(double));
	//cudaMemcpyFromSymbol(points_within_circle_array, dev_points_within_circle_array, ITERATION * sizeof(int));
	//cudaMemcpyFromSymbol(points_within_square_array, dev_points_within_square_array, ITERATION * sizeof(int));
	//cudaMemcpyFromSymbol(pi_array, dev_pi_array, ITERATION * sizeof(double));

	//timer.StopTimer();
	////PrintOut();
	//float elapsedTime = timer.GetTimeSec();
	//std::cout << elapsedTime << std::endl;

	#pragma endregion

	#pragma region N Threads With Blocks

	//int block_count = (ITERATION - 1) / BLOCK_SIZE + 1;
	//EstimatePiNWithBlocks << < block_count, BLOCK_SIZE >> > ();

	//cudaMemcpyFromSymbol(random_x_array, dev_random_x_array, ITERATION * sizeof(double));
	//cudaMemcpyFromSymbol(random_y_array, dev_random_y_array, ITERATION * sizeof(double));
	//cudaMemcpyFromSymbol(points_within_circle_array, dev_points_within_circle_array, ITERATION * sizeof(int));
	//cudaMemcpyFromSymbol(points_within_square_array, dev_points_within_square_array, ITERATION * sizeof(int));
	//cudaMemcpyFromSymbol(pi_array, dev_pi_array, ITERATION * sizeof(double));

	//timer.StopTimer();

	////PrintOut();
	//float elapsedTime = timer.GetTimeSec();
	//std::cout << elapsedTime << std::endl;

	#pragma endregion
	
	#pragma region Shared Memory

	int block_count = (ITERATION - 1) / BLOCK_SIZE + 1;
	EstimatePiShared << < block_count, BLOCK_SIZE >> > ();
	cudaMemcpyFromSymbol(random_x_array, dev_random_x_array, ITERATION * sizeof(double));
	cudaMemcpyFromSymbol(random_y_array, dev_random_y_array, ITERATION * sizeof(double));
	cudaMemcpyFromSymbol(points_within_circle_array, dev_points_within_circle_array, ITERATION * sizeof(int));
	cudaMemcpyFromSymbol(points_within_square_array, dev_points_within_square_array, ITERATION * sizeof(int));
	cudaMemcpyFromSymbol(pi_array, dev_pi_array, ITERATION * sizeof(double));

	timer.StopTimer();

	//PrintOut();
	float elapsedTime = timer.GetTimeSec();
	std::cout << elapsedTime << std::endl;

	#pragma endregion

	#pragma region Shared Memory 2

	//int block_count = (ITERATION - 1) / BLOCK_SIZE + 1;
	//EstimatePiShared2 << < block_count, BLOCK_SIZE >> > ();
	//cudaMemcpyFromSymbol(random_x_array, dev_random_x_array, ITERATION * sizeof(double));
	//cudaMemcpyFromSymbol(random_y_array, dev_random_y_array, ITERATION * sizeof(double));
	//cudaMemcpyFromSymbol(points_within_circle_array, dev_points_within_circle_array, ITERATION * sizeof(int));
	//cudaMemcpyFromSymbol(points_within_square_array, dev_points_within_square_array, ITERATION * sizeof(int));
	//cudaMemcpyFromSymbol(pi_array, dev_pi_array, ITERATION * sizeof(double));

	//timer.StopTimer();

	//PrintOut();
	//float elapsedTime = timer.GetTimeSec();
	//std::cout << elapsedTime << std::endl;

	#pragma endregion
}

