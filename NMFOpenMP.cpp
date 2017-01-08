#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include <climits>
#include <iomanip>
#include <sys/time.h> 
using namespace std;

// void generateWH(float *a, int r, int c, int k, float *w, float *h)
// {
// 	int maxA = 0;
// 	for (int i = 0; i < r; ++i) { //find largest element in matrix //25 in this case
// 		for (int j = 0; j < c; ++j) {
// 			if (a[i * c + j] > maxA) {
// 				maxA = a[i * c + j];
// 			}
// 		}
// 	}

// 	float mx = sqrt(maxA / k); // max for when generating random numbers

// 	// W = r by k
// 	for (int i = 0; i < r; ++i) {
// 		for (int j = 0; j < k; ++j) {
// 			w[i * k + j] = ((float)rand() / (float)(RAND_MAX)) * mx; // generate random init for Matrix W
// 		}
// 	}

// 	// H = k by c
// 	for (int i = 0; i < k; ++i) {
// 		for (int j = 0; j < c; ++j) {
// 			h[i * c + j] = ((float)rand() / (float)(RAND_MAX)) * mx; // generate random init for Matrix W
// 		}
// 	}
// }

void transpose(float *matrix, int row, int col, float *result) {
	#pragma omp parallel for
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			result[j * row + i] = matrix[i * col + j];
		}
	}
}

void matrixMult(float *left, int leftRow, int leftCol, float *right, int rightRow, int rightCol, float *result, int resultRow, int resultCol) {
	#pragma omp parallel for
	for (int i = 0; i < leftRow; ++i) {
		for (int j = 0; j < rightCol; ++j) {
			// you need k calcs for each entry
			for (int k = 0; k < rightRow; ++k) {
				result[i * resultCol + j] += left[i * leftCol + k] * right[k * rightCol + j];
			}
		}
	}
}

void matrixDiv(float *left, float *right, int row, int col, float *result) {
	#pragma omp parallel for
	for (int i = 0; i < row * col; ++i) {
		if (right[i] == 0) {
			result[i] = 1.0e-10;
		}
		else {
			result[i] = left[i] / right[i];
		}
	}
}

void matrixDot(float *left, float *right, int row, int col, float *result) {
	#pragma omp parallel for
	for (int i = 0; i < row * col; ++i) {
		result[i] = left[i] * right[i];
	}
}

void initializeArray(float *arr, int row, int col) {
	#pragma omp parallel for
	for (int i = 0; i < row * col; ++i) {
		arr[i] = 0;
	}
}

// void printMatrix(float *matrix, int row, int col) {
// 	cout.precision(6);
// 	for (int i = 0; i < row; ++i) {
// 		for (int j = 0; j < col; j++){
// 			cout << setw(13) << matrix[i * col + j];
// 		}
// 		cout << endl;
// 	}
// 	cout << endl;
// }

void nmfomp(float *a, int r, int c, int k, int niters, float *w, float *h) {

	generateWH(a, r, c, k, w, h);

	for (int iter = 0; iter < niters; ++iter) {
		//update W
		//W_new = W_old ◦ (A H_old') / (W_old H_old H_old')
		
		float Ht[c * k];
		initializeArray(Ht, c, k);
		transpose(h, k, c, Ht); // transpose pass in row and col of original matrix

		float A_Ht[r * k];
		initializeArray(A_Ht, r, k);
		matrixMult(a, r, c, Ht, c, k, A_Ht, r, k);

		float W_H[r * c];
		initializeArray(W_H, r, c);
		matrixMult(w, r, k, h, k, c, W_H, r, c);

		float WH_Ht[r * k];
		initializeArray(WH_Ht, r, k);
		matrixMult(W_H, r, c, Ht, c, k, WH_Ht, r, k);

		float A_Ht_d_WH_Ht[r * k];
		initializeArray(A_Ht_d_WH_Ht, r, k);
		matrixDiv(A_Ht, WH_Ht, r, k, A_Ht_d_WH_Ht);

		float new_W[r * k];
		initializeArray(new_W, r, k);
		matrixDot(w, A_Ht_d_WH_Ht, r, k, new_W);

		// update H
		//H_new = H_old ◦ (W_old' A) / (W_old' W_old H_old)

		float Wt[k * r];
		initializeArray(Wt, k, r);
		transpose(w, r, k, Wt); // transpose pass in row and col of original matrix

		float Wt_A[k * c];
		initializeArray(Wt_A, k, c);
		matrixMult(Wt, k, r, a, r, c, Wt_A, k, c);

		float Wt_W[k * k];
		initializeArray(Wt_W, k, k);
		matrixMult(Wt, k, r, w, r, k, Wt_W, k, k);

		float WtW_H[k * c];
		initializeArray(WtW_H, k, c);
		matrixMult(Wt_W, k, k, h, k, c, WtW_H, k, c);

		float Wt_A_d_WtW_H[k * c];
		initializeArray(Wt_A_d_WtW_H, k, c);
		matrixDiv(Wt_A, WtW_H, k, c, Wt_A_d_WtW_H);

		float new_H[k * c];
		initializeArray(new_H, k, c);
		matrixDot(h, Wt_A_d_WtW_H, k, c, new_H);

		memcpy(w, new_W, sizeof(int)* r * k);
		memcpy(h, new_H, sizeof(int)* k * c);
	}
	float newA[r * c];
	initializeArray(newA, r, c);
	matrixMult(w, r, k, h, k, c, newA, r, c);
	memcpy(a, newA, sizeof(int)* r * c);
}

// int main() {
// 	float a[2000];
// 	int r = 40;
// 	int c = 50;
// 	int k = 7;
// 	int niters = 1000;
// 	float w[r*k];
// 	float h[k*c];
// 	int num_trials = 100;
// 	struct timeval t1, t2;
// 	double elapsedTime;
// 	double total_time = 0;
// 	for(int i = 0; i < num_trials; i++)
// 	{
// 	generateA(a, r, c);
// 	if(i == num_trials-1)
// 		printMatrix(a, r, c);
// 	gettimeofday(&t1, NULL);
// 	nmfomp(a, r, c, k, niters, w, h);
// 	gettimeofday(&t2, NULL);
// 	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
// 	total_time += elapsedTime;
// 	}
// 	double average = total_time / num_trials;
// 	printMatrix(a, r, c);
// 	cout << "For " << num_trials << " Trials " << "average per trial is " << average << endl;
// 	cout << endl;
// 	return 0;
// }