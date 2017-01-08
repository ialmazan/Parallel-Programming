#include <stdio.h>
//CUDA comparison
void printMatrix(float *matrix, int row, int col) {
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; j++){
      printf("%f ", matrix[i * col + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void transpose(float *matrix, int row, int col, float *result){
  int i = 0;
  int j = 0;
  #pragma acc kernels loop independent copyin(matrix[0:row*col]), copyout(result[0:row*col])
  for (i=0; i < row; ++i) {
    for (j=0; j < col; ++j) {
      result[j * row + i] = matrix[i * col + j];
    }
  }
}

void matrixMult(float *left, int leftRow, int leftCol, float *right, int rightRow, 
                int rightCol, float *result, int resultRow, int resultCol) {
  int i,j,k;
  float temp;

  #pragma acc kernels copyin(left[0:leftRow*leftCol], right[0:rightRow*rightCol]), copyout(result[0:resultRow*resultCol])
  {
    #pragma acc loop independent
    for (i = 0; i < leftRow; ++i) {
      #pragma acc loop independent
      for (j = 0; j < rightCol; ++j) {
        temp=0;
        // you need k calcs for each entry
        for (k = 0; k < rightRow; ++k) {
          temp += left[i * leftCol + k] * right[k * rightCol + j];
        }
        result[i * resultCol + j] = temp;
      }
    }
  }
}

int main(void) {
  int row;
  int col;
  row=3;
  col=3;
  //generate matrix
  float A[3*3] = {1, 2, 3,
                  4, 5, 6,
                  7, 8, 9};
  //vector u
  float u[3*1] = {2,
                  4,
                  6};
  //vector u transposed
  float ut[1*3];
  transpose(u, row, 1, ut);

  //quadratic form
  float result[1];
  //u transpose * A
  float ut_A[1*3];

  matrixMult(ut, 1, col, A, row, col, ut_A, 1, col);
  printMatrix(ut_A, 1, 3);

  matrixMult(ut_A, 1, col, u, row, 1, result, 1, 1);
  printMatrix(result, 1, 1);
  return 0;
}
