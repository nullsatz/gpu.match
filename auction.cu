#include<iostream>
#include"auction.h"

#define NTHREADS 16 // 256

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> d_benefits;

// each thread: an object
// for that object, look for the highest bid from unassigned person
__global__ void AuctionGPU_Assignment(int * d_numAssign, const int n,
	int * I, int * O, float * bids, float * p)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if(j >= n) return;

	float
		tempBid = -1.0, highestBid = 0.0;
	int bidPerson = -1;
	
	//loop over people
	for(int i = 0; i < n; i++) {
		tempBid = bids[i * n + j];
		if(tempBid > highestBid) {
				highestBid = tempBid;
				bidPerson = i;
		}
	}

	if(bidPerson < 0) return;

	// the object j reviews the bid only if
	// bid person != currently assigned person
	if(O[j] == bidPerson) return;

	//unassign the person that was previously assigned to j:
	if(O[j] >= 0) I[O[j]] = -1;

	//raise the price to the winning bid
	//bidsRow = (float *) ((char *)bids + bidPerson * bidsPitch);
	//p[j] = bidsRow[j];
	p[j] = highestBid;

	//assign j to i
	I[bidPerson] = j;
	O[j] = bidPerson;
}

// each unassigned person i finds object j that offers max value to bid on
// each person may bid on any object
__global__ void AuctionGPU_Bidding(const int n, float * bids, float * p,
	int * I, const float  e, int * d_numAssign)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= n) return;
	if(I[i] != -1) return; //unassigned?
	
	//has Unassigned person 
	*d_numAssign = 1;

	// init the 2nd max object value with very low value
	// for the case when the person is only interested in one object
	int
		fir_maxObj = 0;
	float
		sec_maxObjValue = -1000.0, temp_ObjValue = 0.0;

	// float fir_maxObjValue = a[i * n] - p[0];
	float fir_maxObjValue = tex2D(d_benefits, 0, i) - p[0];
	for(int j = 1; j < n; j++) {
		temp_ObjValue = tex2D(d_benefits, j, i) - p[j];
		
		//if is higher that the highest
		if(temp_ObjValue > fir_maxObjValue) {
			sec_maxObjValue = fir_maxObjValue;

			fir_maxObj = j;
			fir_maxObjValue = temp_ObjValue;
		} else if(temp_ObjValue > sec_maxObjValue) {
			//or if is higher that the second highest
			sec_maxObjValue = temp_ObjValue;
		}
	}
	// bidding inc from person i for favorite object
	bids[i * n + fir_maxObj] = fir_maxObjValue - sec_maxObjValue + e;
}

void cudaTimerStart(cudaEvent_t &start) {
	cudaEventCreate(&start);
	cudaEventRecord(start, 0 );
}

float cudaTimerEnd(cudaEvent_t &start) {
	cudaEvent_t stop;
	cudaEventCreate(&stop);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );

	float time;
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	return time;
}

int * d_auction(int cSize, float * h_a) {
	float e = 1.0;

	// Allocating CPU memory 
	// 'h_' prefix - CPU (host) memory space:

	// a[i,j] : desire of person i for object j
	// float * h_a = 0;		
	// h_a = (float *) malloc(sizeof(float) * C_MAX_INSTANCE * C_MAX_INSTANCE);

	// Allocating GPU memory
	// 'd_' prefix - GPU (device) memory space

	// Pick which CUDA capable device to run on
	cudaSetDevice(0);
	// currently set to 0 which would be the default dive
	// if cudaSetDevice weren't called at all

	cudaEvent_t start;
	cudaTimerStart(start);

	//a [i,j] : desire of person i for object j
	// float * d_a;
	// cudaMalloc((void **) & d_a, sizeof(float) * cSize * cSize);

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
		cudaChannelFormatKindFloat);
	cudaArray * cuArray;
	cudaMallocArray(&cuArray, &channelDesc, cSize, cSize);

	// Copy to device memory some data located at address h_data
	// in host memory 
	cudaMemcpyToArray(cuArray, 0, 0, h_a, sizeof(float) * cSize * cSize,
		cudaMemcpyHostToDevice);

	// Set texture reference parameters
	d_benefits.addressMode[0] = cudaAddressModeClamp;
	d_benefits.addressMode[1] = cudaAddressModeClamp;
	d_benefits.filterMode = cudaFilterModePoint;
	d_benefits.normalized = false;

	// Bind the array to the texture reference
	cudaBindTextureToArray(d_benefits, cuArray, channelDesc);

	//bids value
	float * d_bids;
	cudaMalloc(&d_bids, sizeof(float) * cSize * cSize);

	//p[j] : each object j has a price:
	float * d_p;
	cudaMalloc((void **) & d_p, sizeof(float) * cSize);

	//each person is or not assigned
	int * d_i;
	cudaMalloc((void **) & d_i, sizeof(int) * cSize);

	//each object is or not assigned
	int * d_o;
	cudaMalloc((void **) & d_o, sizeof(int) * cSize);

	// used as a boolean that is set whenever there is an unassigned person
	int * d_numAssign;
	cudaMalloc((void **) & d_numAssign, sizeof(int));
	
	dim3 dimBlock(NTHREADS, 1, 1);
	int gx = ceil(cSize /(double) dimBlock.x);
	dim3 dimGrid(gx, 1, 1);

	// copying input data to GPU mem and cleaning aux arrays.
	// in the case of the matrix could go mem constant:
	// so need to clear or copy the size you will use in interaction
	// cudaMemcpy(d_a, h_a, sizeof(float) * cSize * cSize,
	// cudaMemcpyHostToDevice);
	
	// cleaning/initializing algoritm mem
	cudaMemset(d_bids, 0, cSize * cSize * sizeof(float));
	cudaMemset(d_p, 0, cSize * sizeof(float));
	cudaMemset(d_i, -1, cSize * sizeof(int));
	cudaMemset(d_o, -1, cSize * sizeof(int));
	cudaMemset(d_numAssign, 0, sizeof(int));

	int * h_numAssign;
	cudaMallocHost((void **) & h_numAssign, sizeof(int));
	*h_numAssign = 1;

	while(*h_numAssign > 0) {
		cudaMemset(d_bids, 0, cSize * cSize * sizeof(float));
		cudaMemset(d_numAssign, 0, sizeof(int));

		AuctionGPU_Bidding<<<dimBlock, dimGrid>>>(cSize, d_bids, d_p, d_i, e,
			d_numAssign);
            
		cudaMemcpy(h_numAssign, d_numAssign, sizeof(int),
			cudaMemcpyDeviceToHost);

		if(*h_numAssign > 0) {
			AuctionGPU_Assignment<<<dimBlock, dimGrid>>>(d_numAssign, cSize,
				d_i, d_o, d_bids, d_p);							
		}
	}
	cudaFreeHost(h_numAssign);
	cudaFree(d_numAssign);

	// Release GPU memory
    // cudaFree(d_a);
	cudaFreeArray(cuArray);
    cudaFree(d_bids);
    cudaFree(d_p);
	cudaFree(d_o);

	// person assignment results (contain the object number or -1 if unassigned)
	int * h_i_GPUresults = (int *) malloc(sizeof(int) * cSize);

	// Read back GPU results: read the assignements from d_I
	cudaMemcpy(h_i_GPUresults, d_i, sizeof(int) * cSize,
		cudaMemcpyDeviceToHost);
 
	cudaFree(d_i);

	float time = cudaTimerEnd(start);
	std::cout << cSize << " " << time << std::endl;

	return h_i_GPUresults;
}
