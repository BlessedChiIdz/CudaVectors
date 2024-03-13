#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
// Size of array



// Kernel
__device__ double globalArr[4][3];


__global__ void Kernel() {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%f ", globalArr[i][j]);
		}
		printf("\n");
	}
}
// Main program
int main()
{
	
	size_t bytes = 3 * 4 * sizeof(double);
	double TwoDim[3][4];

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			TwoDim[i][j] = rand() % 10 + 1;
		}
	}

	double TwoDimD[4][3];

	cudaMemcpy(globalArr, TwoDim, bytes,cudaMemcpyHostToDevice);
	
	

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Launch kernel
	
	Kernel << < 1, 1 >> > ();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); // Sync events (BLOCKS till last (stop in this case) has been recorded!)
	float elapsedTime; // Initialize elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	
	return 0;
}