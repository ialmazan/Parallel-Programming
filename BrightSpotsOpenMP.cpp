#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>


using namespace std;
/*
void generateA(float *a, int r, int c){
	for (int i = 0; i < r; i++){
		for (int j = 0; j < c; j++){
			a[i*c + j] = ((float)rand() / (float)(RAND_MAX)) * 1;
		}
	}
}
*/

int brights(float *pix, int n, int k, float thresh){
	int brightSpots = 0;
	#pragma omp parallel
	{
	    //for loop control
		#pragma omp for
	    for (int i = 0; i <= n - k; ++i) {//rows
			for (int j = 0; j <= n - k; ++j){//columns
				bool brokeInner = false;
				if(pix[i * n + j] >= thresh){ //no out of bounds
				    for (int l = 0; l < k; ++l){// sub matrix rows
				    	if(brokeInner == true) //if out for not broken continue
						    	break;
						for (int p = 0; p < k; ++p){//sub matric cols
						    if(pix[(i+l) * n + (j+p)] < thresh) // check if neigbor doesn't meet threshold
						    { 
						    	brokeInner = true; //set to break out of outer for, since inner broke no need to check more values
						    	break;
						    }
						    // cout << pix[(i+l) * n + (j+p)] <<" "; //print out submatrix iteration until done or break.
				    	}
				    	// cout << endl;
				    }
				    if(brokeInner == true) //not a full sub matrix of size k
				    	brokeInner = false; // reset for next iteration
				    else // no breaks so found a size k brightspot
				    	#pragma omp critical
				    	{
				    		brightSpots++;
						}
				    // cout << endl;
				}
			}
		}
	}
	return brightSpots; //return count
}

/*
int brightsno(float *pix, int n, int k, float thresh){
	int brightSpots = 0;
	    for (int i = 0; i <= n - k; ++i) {//rows
			for (int j = 0; j <= n - k; ++j){//columns
				bool brokeInner = false;
				if(pix[i * n + j] >= thresh){ //no out of bounds
				    for (int l = 0; l < k; ++l){// sub matrix rows
				    	if(brokeInner == true) //if out for not broken continue
						    	break;
						for (int p = 0; p < k; ++p){//sub matric cols
						    if(pix[(i+l) * n + (j+p)] < thresh) // check if neigbor doesn't meet threshold
						    { 
						    	brokeInner = true; //set to break out of outer for, since inner broke no need to check more values
						    	break;
						    }
						    // cout << pix[(i+l) * n + (j+p)] <<" "; //print out submatrix iteration until done or break.
				    	}
				    	// cout << endl;
				    }
				    if(brokeInner == true) //not a full sub matrix of size k
				    	brokeInner = false; // reset for next iteration
				    else // no breaks so found a size k brightspot
				    		brightSpots++;
				    // cout << endl;
				}
			}
		}
	return brightSpots; //return count
}
*/

/*
void printMatrix(float *matrix, int n) { //print matrix
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; j++){
			cout << matrix[i * n + j]<< " ";
		}
		cout << endl;
	}
	cout << endl;
}
*/

/*
int main() {
	int n = 5000; //dimensino nxn
	float *pix = new float[n*n];
	generateA(pix, n, n);
	//printMatrix(pix, n);
	int k = 15; //brightspots of size kxk
	float thresh = 0.01; //pixel threshold
	int count = 0; // #of brightspots
	struct timeval t1, t2;
	double elapsedTime;
	gettimeofday(&t1, NULL);
	
	count = brightsno(pix, n, k, thresh); // call finder
	cout <<"There are " << count <<" brightSpots in non para" << endl; //result

	//count = brights(pix, n, k, thresh); // call finder
	//cout <<"There are " << count <<" brightSpots in para" << endl; //result
	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
	cout << "Time: " << elapsedTime << endl;

	return 0;
}
*/