#include<iostream>
#include"auction.h"

// #define NTHREADS 256
#define NTHREADS 16

#define TRUE 1
#define FALSE !TRUE

#define e 1

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> d_benefits;

// each thread: an object
// for that object, look for the highest bid from unassigned person
__global__ void AuctionGPU_Assignment(const int nBidders, const int nItems,
	float * bids, float * prices, int * bidderItems, int * itemBidders)
{
	int item = blockDim.x * blockIdx.x + threadIdx.x;
	if(item >= nItems) return;

	float
		tempBid = -1.0, highestBid = 0.0;

	int bidder = -1;
	
	//loop over bidders
	for(int i = 0; i < nBidders; i++) {
		tempBid = bids[i * nItems + item];
		if(tempBid > highestBid) {
				highestBid = tempBid;
				bidder = i;
		}
	}

	if(bidder < 0) return;

	// the object j reviews the bid only if
	// bidder != currently assigned bidder
	if(itemBidders[item] == bidder) return;

	//unassign the person that was previously assigned to j:
	if(itemBidders[item] >= 0)
		bidderItems[itemBidders[item]] = -1;

	//raise the price to the winning bid
	prices[item] = highestBid;

	//assign j to i
	bidderItems[bidder] = item;
	itemBidders[item] = bidder;
}

// each unassigned bidder finds and bids on item j that offers max value
__global__ void AuctionGPU_Bidding(const int nBidders, const int nItems,
	float * bids, float * prices, int * bidderItems, int * anyUnassigned)
{
	int bidder = blockDim.x * blockIdx.x + threadIdx.x;
	if(bidder >= nBidders) return;
	if(bidderItems[bidder] != -1) return; //unassigned?
	
	//has Unassigned person 
	*anyUnassigned = TRUE;

	// init the 2nd max object value with very low value
	// for the case when the person is only interested in one object
	int
		maxItem = 0;
	float
		secondMaxItemValue = -1000.0, tempItemValue = 0.0;

	// float maxItemValue = a[i * n] - p[0];
	float maxItemValue = tex2D(d_benefits, 0, bidder) - prices[0];
	for(int j = 1; j < nItems; j++) {
		tempItemValue = tex2D(d_benefits, j, bidder) - prices[j];
		
		//if is higher that the highest
		if(tempItemValue > maxItemValue) {
			secondMaxItemValue = maxItemValue;

			maxItem = j;
			maxItemValue = tempItemValue;
		} else if(tempItemValue > secondMaxItemValue) {
			//or if is higher that the second highest
			secondMaxItemValue = tempItemValue;
		}
	}
	// bidding inc from person i for favorite object
	bids[bidder * nItems + maxItem] = maxItemValue - secondMaxItemValue + e;
}

void cudaTimerStart(cudaEvent_t &start) {
	cudaEventCreate(&start);
	cudaEventRecord(start, 0 );
}

float cudaTimerStop(cudaEvent_t &start) {
	cudaEvent_t stop;
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return time;
}

// 'h_' prefix - CPU (host) memory space
// 'd_' prefix - GPU (device) memory space

// h_benefits[i, j] : desire of bidder i for item j

int * d_auction(int nBidders, int nItems, float * h_benefits) {

	// Pick a CUDA capable device to run on
	// currently set to 0 which would be the default device
	// if cudaSetDevice weren't called at all
	cudaSetDevice(0);

	// start timing for performance profiling
	cudaEvent_t start;
	cudaTimerStart(start);

	size_t
		matFSize = nBidders * nItems * sizeof(float),
		itemsFSize = nItems * sizeof(float),
		itemsISize = nItems * sizeof(int),
		biddersISize = nBidders * sizeof(int);

	// make a cached read only device texture to store benefits matrix
	// the texture is a global defined at the top of the file
	// textures must be global
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
		cudaChannelFormatKindFloat);
	cudaArray * cuArray;
	cudaMallocArray(&cuArray, &channelDesc, nItems, nBidders);
	cudaMemcpyToArray(cuArray, 0, 0, h_benefits, matFSize,
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
	cudaMalloc(&d_bids, matFSize);

	//price[j] : each item j has a price:
	float * d_prices;
	cudaMalloc((void **) & d_prices, itemsFSize);

	//each bidder is (bidderItems[x] = item index)
	//	or is not assigned (bidderItems[x] = -1)
	int * d_bidderItems;
	cudaMalloc((void **) & d_bidderItems, biddersISize);

	//each object is (itemBidders[x] = bidder index)
	//	or is not assigned (itemBidders[x] = -1)
	int * d_itemBidders;
	cudaMalloc((void **) & d_itemBidders, itemsISize);

	// used as a boolean that is TRUE if there is an item not yet assigned
	int * d_anyUnassigned;
	cudaMalloc((void **) & d_anyUnassigned, sizeof(int));

	dim3 dimBlock(NTHREADS, 1, 1);
	int gx = ceil((double) nItems /(double) dimBlock.x);
	dim3 dimGrid(gx, 1, 1);

	// copying input data to GPU mem and cleaning aux arrays.
	// in the case of the matrix could go mem constant:
	// so need to clear or copy the size you will use in interaction
	// cudaMemcpy(d_a, h_a, sizeof(float) * cSize * cSize,
	// cudaMemcpyHostToDevice);
	
	// cleaning/initializing algoritm mem
	cudaMemset(d_bids, 0, matFSize);
	cudaMemset(d_prices, 0, itemsFSize);
	cudaMemset(d_bidderItems, -1, biddersISize);
	cudaMemset(d_itemBidders, -1, itemsISize);
	cudaMemset(d_anyUnassigned, 0, sizeof(int));

	int * h_anyUnassigned;
	cudaMallocHost((void **) & h_anyUnassigned, sizeof(int));
	*h_anyUnassigned = 1;

	while(*h_anyUnassigned > 0) {
		cudaMemset(d_bids, 0, matFSize);
		cudaMemset(d_anyUnassigned, 0, sizeof(int));

		AuctionGPU_Bidding<<<dimBlock, dimGrid>>>(nBidders, nItems, d_bids,
			d_prices, d_bidderItems, d_anyUnassigned);
            
		cudaMemcpy(h_anyUnassigned, d_anyUnassigned, sizeof(int),
			cudaMemcpyDeviceToHost);

		if(*h_anyUnassigned > 0) {
			AuctionGPU_Assignment<<<dimBlock, dimGrid>>>(nBidders, nItems,
				d_bids, d_prices, d_bidderItems, d_itemBidders);
		}
	}
	cudaFreeHost(h_anyUnassigned);
	cudaFree(d_anyUnassigned);

	// Release GPU memory
    // cudaFree(d_a);
	cudaFreeArray(cuArray);
    cudaFree(d_bids);
    cudaFree(d_prices);
	cudaFree(d_itemBidders);

	// person assignment results (contain the object number or -1 if unassigned)
	int * h_bidderItems = (int *) malloc(biddersISize);

	// Read back GPU results: read the assignements from d_I
	cudaMemcpy(h_bidderItems, d_bidderItems, biddersISize,
		cudaMemcpyDeviceToHost);
 
	cudaFree(d_bidderItems);

	float time = cudaTimerStop(start);
	std::cout << nBidders << " " << nItems << " " << time << std::endl;

	return h_bidderItems;
}
