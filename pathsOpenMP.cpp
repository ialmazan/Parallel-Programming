#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>

using namespace std;
int pathIndex=0;
void fillpaths(int *adjm, int n, int k, int *paths, int vertex, int curlen, int *curPath);
void printArray(int *arr, int len);

// http://www.perlmonks.org/?node_id=522257
// https://www.youtube.com/watch?v=z5iEPa-aFa0

void generateA(float *a, int r, int c){
	for (int i = 0; i < r; i++){
		for (int j = 0; j < c; j++){
			a[i*c + j] = ((float)rand() / (float)(RAND_MAX)) * 1;
		}
	}
}

void printMatrix(int *matrix, int n) { //print matrix
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; j++){
			cout << matrix[i * n + j]<< " ";
		}
		cout << endl;
	}
	cout << endl;
}


void matrixMult(int *left, int *right, int *result, int n) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			// you need k calcs for each entry
			for (int k = 0; k < n; ++k) {
				result[i * n + j] += left[i * n + k] * right[k * n + j];
			}
		}
	}
}


//returns the total number of paths of length k
//k is length of the path
//n is adjm row or col
int findnumpaths(int *adjm, int n, int k) {
	int result[n*n];
	memset(result, 0, sizeof(result));	
	int temp[n*n];
	memcpy(temp, adjm, sizeof(int)*n*n);
		
	//output is stored in "temp"
	//result is clear to 0... every iteration
	for (int i = 1; i < k; ++i) {
		matrixMult(temp, adjm, result, n);
		//printMatrix(result, n); 
		memcpy(temp, result, sizeof(int) * n * n);
		memset(result, 0, sizeof(result));
	}

	int totalPath = 0;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			totalPath += temp[i*n+j];
		}
	}

	return totalPath;

}

//this function is given from HW
void findpaths(int *adjm, int n, int k, int *paths, int *numpaths){
	numpaths = new int();
	*numpaths = findnumpaths(adjm, n, k);
	//add numpaths because 2 edges require 3 nodes to print
	paths = new int[k*(*numpaths) + *numpaths];
	int *ptr;
	//for loop use to loop through all possibilities
	//find all path start from all indexes
	#pragma omp parallel for
	for(int i = 0; i < n; i++){
		fillpaths(adjm, n, k, paths, i, 0, ptr);
	}
//	printArray(paths, k*(*numpaths) + *numpaths);

//	cout << "Number of paths: " << *numpaths << endl;
}

void printArray(int *arr, int len){
	for(int i = 0; i < len; i++){
		cout << arr[i] << " ";
	}
	cout << endl;
}
/* given a vertex, go through its row and recurse */
/*
Example: 2 b 2 matrix index 0 = 1, 1 = 2, 2 = 1, 3 = 0
  1  2
  1  0
if vertex is 0, it will recurse 3 times, 1 time in i = 0 and 2 times in i = 1

*/
void criticalUpdate(int k, int *paths, int *curPath) {
	memcpy(paths+pathIndex, curPath, sizeof(int) * (k + 1) );
	pathIndex += k+1;
}

void fillpaths(int *adjm, int n, int k, int *paths, int vertex, int curlen, int *curPath){
	if(curlen == 0){
		for(int i = 0; i < n; i++){
			//if adjm[i][j] == 2 then we call recursively 2 times
			for(int vToi = adjm[vertex*n+i]; vToi > 0; vToi--){
				//let next loop start with vertex i
				int tempPath[k+1];
				tempPath[0] = vertex;
				fillpaths(adjm, n, k, paths, i, curlen+1, tempPath);
			}
		}
		return;
	}
	//we are done, append curpath to paths
		if(curlen == k){
			/******LOCKS?********/
			//cout << "Appending" << endl;
			curPath[k] = vertex;
			//printArray(curPath, k+1);
			#pragma omp critical
			criticalUpdate(k, paths, curPath);

			return;
		}
		
	//not at the start or end
	//need to keep move on
	for(int i = 0; i < n; i++){
		//if adjm[i][j] == 2 then we call recursively 2 times
		for(int vToi = adjm[vertex*n+i]; vToi > 0; vToi--){
			//let next loop start with vertex i
			curPath[curlen] = vertex;
			fillpaths(adjm, n, k, paths, i, curlen+1, curPath);
		}
	}

}


// int main() {

// 	// this 8x8 adjacency matrix for testing http://i.stack.imgur.com/EDeI1.png
//     int adjm[] = {0,1,0,0,0,0,0,0,
//   				0,0,1,1,1,0,0,0,
//   				0,3,0,0,0,0,1,0,
//   				0,0,0,0,0,1,0,0,
//   				1,0,2,0,0,1,0,1,
//   				0,0,0,0,0,0,1,0,
//   				2,0,0,0,0,0,0,1,
//   				4,3,0,0,0,0,0,0
//           	   };
//   	int n = 8;
//   	int k = 4;
//   	int *paths;
//     //call master function
//   	int *numpaths;

//   	//cout << "input adj matrix" << endl;
//   	//printMatrix(adjm, n);

//     findpaths(adjm, n, k, paths, numpaths);
    
// 	return 0;
// }
