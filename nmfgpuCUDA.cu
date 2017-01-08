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

//https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA
//was looking at above source for matrix mult example

using namespace std;

void generateA(float *a, int r, int c){
    for (int i = 0; i < r; i++){
        for (int j = 0; j < c; j++){
            a[i*c + j] = rand()%100;
        }
    }
}

void generateWH(float *a, int r, int c, int k, float *w, float *h)
{
    int maxA = 0;
    for (int i = 0; i < r; ++i) { //find largest element in matrix //25 in this case
        for (int j = 0; j < c; ++j) {
            if (a[i * c + j] > maxA) {
                maxA = a[i * c + j];
            }
        }
    }

    float mx = sqrt(maxA / k); // max for when generating random numbers

    // W = r by k
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < k; ++j) {
            w[i * k + j] = ((float)rand() / (float)(RAND_MAX)) * mx; // generate random init for Matrix W
        }
    }

    // H = k by c
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < c; ++j) {
            h[i * c + j] = ((float)rand() / (float)(RAND_MAX)) * mx; // generate random init for Matrix W
        }
    }
}

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

__global__ void matrixDiv(float *left, float *right, int row, int col, float *result) {
    int i = blockIdx.x;
    //for (int i = 0; i < row * col; ++i) {
    if(i < row * col){
        if (right[i] == 0) {
            result[i] = 1.0e-10;
        }
        else {
            result[i] = left[i] / right[i];
        }
    }
}

__global__ void matrixDot(float *left, float *right, int row, int col, float *result) {
    // #pragma omp parallel for
    int i = blockIdx.x;
    if(i < row * col){
    //for (int i = 0; i < row * col; ++i) {
        result[i] = left[i] * right[i];
    }
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

void nmfgpu(float *a, int r, int c, int k, int niters, float *w, float *h) {

 	generateWH(a, r, c, k, w, h);

    //float a[] = {1,2,3,4,5,6,7,8,9};
    //cout << "A" <<endl;
    //printMatrix(a,r,c);
    //cout << "W" <<endl;
    //float w[] = {1,4,2,5,3,6};
    //printMatrix(w, r, k);
    //float h[] = {4,5,6,7,8,9};
    //cout << "H" <<endl;
    //printMatrix(h, k, c);

    dim3 dimBlock1(1,1);
    dim3 dimBlock(32, 32);

    float *d__a;
    cudaMalloc((void **)&d__a, r*c*sizeof(float));
    cudaMemcpy(d__a, a, r*c*sizeof(float), cudaMemcpyHostToDevice);

    float *d__w;
    cudaMalloc((void **)&d__w, r*k*sizeof(float));
    cudaMemcpy(d__w, w, k*r*sizeof(float), cudaMemcpyHostToDevice);

    float *d__h;
    cudaMalloc((void **)&d__h, k*c*sizeof(float));
    cudaMemcpy(d__h, h, k*c*sizeof(float), cudaMemcpyHostToDevice);

    for (int iter = 0; iter < niters; ++iter) {
        //update W
        //W_new = W_old ◦ (A H_old') / (W_old H_old H_old')
        
        dim3 dimGridHt(c,1);
        //float h__Ht[c * k];
        //initializeArray(h__Ht, c, k);
        float *d__Ht;
        cudaMalloc((void **)&d__Ht, c*k*sizeof(float));
        //cudaMemcpy(d__Ht, h__Ht, c*k*sizeof(float), cudaMemcpyHostToDevice);
        transpose<<<dimGridHt, dimBlock1>>>(d__h, k, c, d__Ht); // transpose pass in row and col of original matrix
        //cudaMemcpy(h__Ht, d__Ht, c*k*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "Ht" <<endl;
        //printMatrix(h__Ht, c, k);

        dim3 dimGridKR((k + dimBlock.x - 1) / dimBlock.x,(r + dimBlock.y - 1) / dimBlock.y);
        //float h__A_Ht[r * k];
        //initializeArray(h__A_Ht, r, k);
        float *d__A_Ht;
        cudaMalloc((void **)&d__A_Ht, r*k*sizeof(float));
        //cudaMemcpy(d__A_Ht, h__A_Ht, r*k*sizeof(float), cudaMemcpyHostToDevice);
        matrixMult<<<dimGridKR, dimBlock>>>(d__a, r, c, d__Ht, c, k, d__A_Ht, r, k);
        //cudaMemcpy(h__A_Ht, d__A_Ht, c*k*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "A * Ht" <<endl;
        //printMatrix(h__A_Ht, r, k);

        dim3 dimGridCR((c + dimBlock.x - 1) / dimBlock.x,(r + dimBlock.y - 1) / dimBlock.y);
        //float h__W_H[r * c];
        //initializeArray(h__W_H, r, c);
        float *d__W_H;
        cudaMalloc((void **)&d__W_H, r*c*sizeof(float));
        //cudaMemcpy(d__W_H, h__W_H, r*c*sizeof(float), cudaMemcpyHostToDevice);
        matrixMult<<<dimGridCR, dimBlock>>>(d__w, r, k, d__h, k, c, d__W_H, r, c);
        //cudaMemcpy(h__W_H, d__W_H, r*c*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "W * H" <<endl;
        //printMatrix(h__W_H, r, c);

        //dimGrid KR declared already
        //float h__WH_Ht[r * k];
        //initializeArray(h__WH_Ht, r, k);
        float *d__WH_Ht;
        cudaMalloc((void **)&d__WH_Ht, r*k*sizeof(float));
        //cudaMemcpy(d__WH_Ht,  h__WH_Ht, r*k*sizeof(float), cudaMemcpyHostToDevice);
        matrixMult<<<dimGridKR, dimBlock>>>(d__W_H, r, c, d__Ht, c, k, d__WH_Ht, r, k);
        //cudaMemcpy(h__WH_Ht, d__WH_Ht, r*k*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "WH * Ht" <<endl;
        //printMatrix(h__WH_Ht, r, k);

        dim3 dimGridDivDot1(r*k,1);
        //float h__A_Ht_d_WH_Ht[r * k];
        //initializeArray(h__A_Ht_d_WH_Ht, r, k);
        float *d__A_Ht_d_WH_Ht;
        cudaMalloc((void **)&d__A_Ht_d_WH_Ht, r*k*sizeof(float));
        //cudaMemcpy(d__A_Ht_d_WH_Ht,  h__A_Ht_d_WH_Ht, r*k*sizeof(float), cudaMemcpyHostToDevice);
        matrixDiv<<<dimGridDivDot1, dimBlock>>>(d__A_Ht, d__WH_Ht, r, k, d__A_Ht_d_WH_Ht);
        //cudaMemcpy(h__A_Ht_d_WH_Ht, d__A_Ht_d_WH_Ht, r*k*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "A * Ht / WH * Ht" <<endl;
        //printMatrix(h__A_Ht_d_WH_Ht, r, k);

        //dimGrid DivDot1 declared prev
        //float h__new_W[r * k];
        //initializeArray(h__new_W, r, k);
        float *d__new_W;
        cudaMalloc((void **)&d__new_W, r*k*sizeof(float));
        //cudaMemcpy(d__new_W,  h__new_W, r*k*sizeof(float), cudaMemcpyHostToDevice);
        matrixDot<<<dimGridDivDot1, dimBlock>>>(d__w, d__A_Ht_d_WH_Ht, r, k, d__new_W);
        //cudaMemcpy(h__new_W, d__new_W, r*k*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "new W = W ◦ (A * Ht / WH * Ht)" <<endl;
        //printMatrix(h__new_W, r, k);



        // update H
        //H_new = H_old ◦ (W_old' A) / (W_old' W_old H_old)

        dim3 dimGridWt(r,1);
        //float h__Wt[k * r];
        //initializeArray(h__Wt, k, r);
        float *d__Wt;
        cudaMalloc((void **)&d__Wt, k*r*sizeof(float));
        //cudaMemcpy(d__Wt, h__Wt, k*r*sizeof(float), cudaMemcpyHostToDevice);
        transpose<<<dimGridWt, dimBlock1>>>(d__w, r, k, d__Wt); // transpose pass in row and col of original matrix
        //cudaMemcpy(h__Wt, d__Wt, k*r*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "Wt" <<endl;
        //printMatrix(h__Wt, k, r);

        dim3 dimGridCK((c + dimBlock.x - 1) / dimBlock.x,(k + dimBlock.y - 1) / dimBlock.y);
        //float h__Wt_A[k * c];
        //initializeArray(h__Wt_A, k, c);
        float *d__Wt_A;
        cudaMalloc((void **)&d__Wt_A, k*c*sizeof(float));
        //cudaMemcpy(d__Wt_A,  h__Wt_A, k*c*sizeof(float), cudaMemcpyHostToDevice);
        matrixMult<<<dimGridCK, dimBlock>>>(d__Wt, k, r, d__a, r, c, d__Wt_A, k, c);
        //cudaMemcpy(h__Wt_A, d__Wt_A, k*c*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "Wt * A" <<endl;
        //printMatrix(h__Wt_A, k, c);

        dim3 dimGridKK((k + dimBlock.x - 1) / dimBlock.x,(k + dimBlock.y - 1) / dimBlock.y);
        //float h__Wt_W[k * k];
        //initializeArray(h__Wt_W, k, k);
        float *d__Wt_W;
        cudaMalloc((void **)&d__Wt_W, k*k*sizeof(float));
        //cudaMemcpy(d__Wt_W,  h__Wt_W, k*k*sizeof(float), cudaMemcpyHostToDevice);
        matrixMult<<<dimGridKK, dimBlock>>>(d__Wt, k, r, d__w, r, k, d__Wt_W, k, k);
        //cudaMemcpy(h__Wt_W, d__Wt_W, k*k*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "Wt * W" <<endl;
        //printMatrix(h__Wt_W, k, k);

        //dimGrid CK declared already
        //float h__WtW_H[k * c];
        //initializeArray(h__WtW_H, k, c);
        float *d__WtW_H;
        cudaMalloc((void **)&d__WtW_H, k*c*sizeof(float));
        //cudaMemcpy(d__Wt_W,  h__WtW_H, k*c*sizeof(float), cudaMemcpyHostToDevice);
        matrixMult<<<dimGridKK, dimBlock>>>(d__Wt_W, k, k, d__h, k, c, d__WtW_H, k, c);
        //cudaMemcpy(h__WtW_H, d__WtW_H, k*c*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "WtW * H" <<endl;
        //printMatrix(h__WtW_H, k, c);

        dim3 dimGridDivDot2(k*c,1);
        //float h__Wt_A_d_WtW_H[k * c];
        //initializeArray(h__Wt_A_d_WtW_H, k, c);
        float *d__Wt_A_d_WtW_H;
        cudaMalloc((void **)&d__Wt_A_d_WtW_H, k*c*sizeof(float));
        //cudaMemcpy(d__Wt_A_d_WtW_H,  h__Wt_A_d_WtW_H, k*c*sizeof(float), cudaMemcpyHostToDevice);
        matrixDiv<<<dimGridDivDot2, dimBlock>>>(d__Wt_A, d__WtW_H, k, c, d__Wt_A_d_WtW_H);
        //cudaMemcpy(h__Wt_A_d_WtW_H, d__Wt_A_d_WtW_H, k*c*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "Wt * A / WtW * H" <<endl;
        //printMatrix(h__Wt_A_d_WtW_H, k, c);

        //dimGrid DivDot2 declared prev
        //float h__new_H[k * c];
        //initializeArray(h__new_H, k, c);
        float *d__new_H;
        cudaMalloc((void **)&d__new_H, k*c*sizeof(float));
        //cudaMemcpy(d__new_H,  h__new_H, k*c*sizeof(float), cudaMemcpyHostToDevice);
        matrixDot<<<dimGridDivDot2, dimBlock>>>(d__h, d__Wt_A_d_WtW_H, k, c, d__new_H);
        //cudaMemcpy(h__new_H, d__new_H, k*c*sizeof(float), cudaMemcpyDeviceToHost);
        //cout << "new W = H ◦ (Wt * A / WtW * H)" <<endl;
        //printMatrix(h__new_H, k, c);

        cudaMemcpy(d__w, d__new_W, r*k*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(d__h, d__new_H, k*c*sizeof(float), cudaMemcpyDeviceToHost);
        /*
        memcpy(w, new_W, sizeof(int)* r * k);
        memcpy(h, new_H, sizeof(int)* k * c);*/

    }
	dim3 dimGridCR((c + dimBlock.x - 1) / dimBlock.x,(r + dimBlock.y - 1) / dimBlock.y);
    //float h__newA[r * c];
    //initializeArray(h__newA, r, c);
    float *d__newA;
    cudaMalloc((void **)&d__newA, r*c*sizeof(float));
    //cudaMemcpy(d__newA,  h__newA, r*c*sizeof(float), cudaMemcpyHostToDevice);
    matrixMult<<<dimGridCR, dimBlock>>>(d__w, r, k, d__h, k, c, d__newA, r, c);
    cudaMemcpy(a, d__newA, r*c*sizeof(float), cudaMemcpyDeviceToHost);
}

// int main() {
//     float a[] = {1,2,3,4,5,6,7,8,9};
//     int r = 3;
//     int c = 3;
//     int k = 2;
//     int niters = 10;
//     float w[r*k];
//     float h[k*c];
//     //generateA(a, r, c);
//     nmfgpu(a, r, c, k, niters, w, h);
//     cout << endl << "CUDA NMF" <<endl;
//     printMatrix(a, r, c);
//     cout << endl;

//     return 0;
// }