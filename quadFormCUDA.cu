#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include <climits>
#include <iomanip>
#include <sys/time.h> 
#include <cuda.h>

using namespace std;

__global__ void transpose(float *matrix, int row, int col, float *result) {
    int i = blockIdx.x;
    if(i < row){
    	for (int j = 0; j < col; ++j) {
    		result[j * row + i] = matrix[i * col + j];
    	}
    }
    
}


__global__ void matrixMult(float *left, int leftRow, int leftCol, float *right, int rightRow, int rightCol, float *result, int resultRow, int resultCol) {
    //#pragma omp parallel for
    float temp = 0;
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if ((i < leftRow) && (j < rightCol)){
    //for (int i = 0; i < leftRow; ++i) {
        //for (int j = 0; j < rightCol; ++j) {
            // you need k calcs for each entry
            for (int k = 0; k < leftCol; ++k) {
                temp += left[i * leftCol + k] * right[k * rightCol + j];
            }
        result[i * resultCol + j] = temp;
        }
        //}
    //}
}

void initializeArray(float *arr, int row, int col) {
    // #pragma omp parallel for
    for (int i = 0; i < row * col; ++i) {
        arr[i] = 0;
    }
}

void printMatrix(float *matrix, int row, int col) {
    cout.precision(6);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; j++){
            cout << setw(13) << matrix[i * col + j];
        }
        cout << endl;
    }
    cout << endl;
}

float gpuquad(float *a, int n, float *u) {

    dim3 dimBlock(32, 32);

    float *d__a;
    cudaMalloc((void **)&d__a, n*n*sizeof(float));
    cudaMemcpy(d__a, a, n*n*sizeof(float), cudaMemcpyHostToDevice);
    //cout << "A" << endl;
    //printMatrix(a, n, n);

    float *d__u;
    cudaMalloc((void **)&d__u, n*sizeof(float));
    cudaMemcpy(d__u, u, n*sizeof(float), cudaMemcpyHostToDevice);
    //cout << "U" << endl;
    //printMatrix(u,n,1);

	dim3 dimGridUt(1,n);
	//float h__Ut[n];
    //initializeArray(h__Ut, 1, n);
    float *d__Ut;
    cudaMalloc((void **)&d__Ut, n*sizeof(float));
    //cudaMemcpy(d__Ut, h__Ut, n*sizeof(float), cudaMemcpyHostToDevice);
    transpose<<<dimGridUt, dimBlock>>>(d__u, 1, n, d__Ut); // transpose pass in row and col of original matrix
    //cudaMemcpy(h__Ut, d__Ut, n*sizeof(float), cudaMemcpyDeviceToHost);
    //cout << "Ut" <<endl;
    //printMatrix(h__Ut, 1, n);

    dim3 dimGridN1((n + dimBlock.x - 1) / dimBlock.x,(1 + dimBlock.y - 1) / dimBlock.y);
    //float h__Ut_a[n];
    //initializeArray(h__Ut_a, 1, n);
    float *d__Ut_a;
    cudaMalloc((void **)&d__Ut_a, n*sizeof(float));
    //cudaMemcpy(d__Ut_a, h__Ut_a, n*sizeof(float), cudaMemcpyHostToDevice);
    matrixMult<<<dimGridN1, dimBlock>>>(d__Ut, 1, n, d__a, n, n, d__Ut_a, 1, n);
    //cudaMemcpy(h__Ut_a, d__Ut_a, n*sizeof(float), cudaMemcpyDeviceToHost);
    //cout << "Ut * A" <<endl;
    //printMatrix(h__Ut_a, 1, n);

    dim3 dimGrid11((1 + dimBlock.x - 1) / dimBlock.x,(1 + dimBlock.y - 1) / dimBlock.y);
    float h__Ut_a_U[1];
    //h__Ut_a_U[0] = 0;
    float *d__Ut_a_U;
    cudaMalloc((void **)&d__Ut_a_U, sizeof(float));
    //cudaMemcpy(d__Ut_a_U, h__Ut_a_U, sizeof(float), cudaMemcpyHostToDevice);
    matrixMult<<<dimGrid11, dimBlock>>>(d__Ut_a, 1, n, d__u, n, 1, d__Ut_a_U, 1, 1);
    cudaMemcpy(h__Ut_a_U, d__Ut_a_U, sizeof(float), cudaMemcpyDeviceToHost);
    //cout << "Ut * A * U" <<endl;
    //printMatrix(h__Ut_a_U, 1, 1);
    return h__Ut_a_U[0];
}
/*
int main() {
    float a[] = {1,2,2,2,3,
    			 2,1,5,5,2,
    			 2,5,1,5,2,
    			 2,5,5,1,2,
    			 3,2,2,2,1};
    int n = 5;
    float u[] = {1,2,3,4,5};
    cout << gpuquad(a, n, u) << endl;
    gpuquad(a, n, u);

    return 0;
}
*/