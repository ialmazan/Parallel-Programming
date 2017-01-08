#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

void generateA(float *a, int r, int c){
	for (int i = 0; i < r; i++){
		for (int j = 0; j < c; j++){
			a[i*c + j] = 1;//((float)rand() / (float)(RAND_MAX)) * 1;
		}
	}
}
//bright spots
int brights(float *pix, int n, int k, float thresh){
	int brightSpots = 0;
	int count = 0;
	int i,j,l,p;

	// #pragma acc data copy(pix[0:n*n])
	// #pragma acc parallel
	// {
	// 	#pragma acc loop reduction(+:brightSpots)
	    for (i = 0; i <= n - k; ++i) {//rows

			for (j = 0; j <= n - k; ++j){//columns
				
				int brokeInner = 0;
				if(pix[i * n + j] >= thresh){ //no out of bounds
				    for (l = 0; l < k; ++l){// sub matrix rows
				    	if(brokeInner == 1) //if out for not broken continue
						    	break;
						for (p = 0; p < k; ++p){//sub matric cols
						    if(pix[(i+l) * n + (j+p)] < thresh) // check if neigbor doesn't meet threshold
						    { 
						    	brokeInner = 1; //set to break out of outer for, since inner broke no need to check more values
						    	break;
						    }
						    // cout << pix[(i+l) * n + (j+p)] <<" "; //print out submatrix iteration until done or break.
				    	}
				    	// cout << endl;
				    }
				    if(brokeInner == 1) //not a full sub matrix of size k
				    {
				    	brokeInner = 0; // reset for next iteration
				    }
				    else // no breaks so found a size k brightspot
				    {
				    	// #pragma acc update device(brightSpots)
				    	// #pragma acc atomic update
				    	// {
				    		brightSpots++;
				    	// }
				    }

				}
			}
		}
	// }

			
	// // //printf("Count is %d\n", count);
	// #pragma acc data copy(count)
	// 	#pragma acc parallel
	// 	{
	// 		#pragma acc loop
	// 		for(i = 0; i < n; i++){
	// 			// printf("X:%d  ",i);
				
	// 			for(j = 0; j < n; j++){
	// 				#pragma acc atomic
	// 				{
	// 					count++;
	// 				}
	// 			}
	// 		}
	// 	}
	// 	printf("count is %d", count);
	return brightSpots; //return count
}

int main() {
	int n = 500; //dimensino nxn
	float *pix = (float*)malloc(sizeof(float)*n*n);
	generateA(pix, n, n);
	//printMatrix(pix, n);
	int k = 2; //brightspots of size kxk
	float thresh = 0.01; //pixel threshold
	int count = 0; // #of brightspots

	
	count = brights(pix, n, k, thresh); // call finder
	printf("There are %d brightSpots.\n", count);
	return 0;
}