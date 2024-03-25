#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#define BLOCK_DIM 1
__shared__ float temp[BLOCK_DIM][BLOCK_DIM];

__global__ void transposeMatrixFast(float* inputMatrix, float* outputMatrix, int width, int height)
{
	

	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height))
	{
		int idx = yIndex * width + xIndex;

		temp[threadIdx.y][threadIdx.x] = inputMatrix[idx];
	}

	__syncthreads();

	xIndex = blockIdx.y * blockDim.y + threadIdx.x;
	yIndex = blockIdx.x * blockDim.x + threadIdx.y;

	if ((xIndex < height) && (yIndex < width))
	{
		int idx = yIndex * height + xIndex;

		outputMatrix[idx] = temp[threadIdx.x][threadIdx.y];
	}
}

__host__ void printMatrixToFile(char* fileName, float* matrix, int width, int height)
{
    FILE* file = fopen(fileName, "wt");
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            fprintf(file, "%f\t", matrix[y * width + x]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}




int main()
{
    int width = 3;   
    int height = 4;

    int matrixSize = width * height;
    int byteSize = matrixSize * sizeof(float);

    float* inputMatrix = new float[matrixSize];
    float* outputMatrix = new float[matrixSize];

    for (int i = 0; i < matrixSize; i++)
    {
        inputMatrix[i] = i;
    }
    int qwe = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", inputMatrix[qwe]);
            qwe++;
        }
        printf("\n");
    }
    
    
        float* devInputMatrix;
        float* devOutputMatrix;

        cudaMalloc((void**)&devInputMatrix, byteSize);
        cudaMalloc((void**)&devOutputMatrix, byteSize);

        cudaMemcpy(devInputMatrix, inputMatrix, byteSize, cudaMemcpyHostToDevice);

        dim3 gridSize = dim3(width / BLOCK_DIM, height / BLOCK_DIM, 1);
        dim3 blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);

        cudaEvent_t start;
        cudaEvent_t stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

       
        
        transposeMatrixFast << <gridSize, blockSize >> > (devInputMatrix, devOutputMatrix, width, height);

        cudaEventRecord(stop, 0);

        float time = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        printf("GPU compute time: %.0f\n", time);

        cudaMemcpy(outputMatrix, devOutputMatrix, byteSize, cudaMemcpyDeviceToHost);

        cudaFree(devInputMatrix);
        cudaFree(devOutputMatrix);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    

    printMatrixToFile("after.txt", outputMatrix, height, width);

    delete[] inputMatrix;
    delete[] outputMatrix;

    return 0;
}