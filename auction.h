#ifndef AUCTION_H
#define AUCTION_H

#ifndef C_MAX_INSTANCE
#define C_MAX_INSTANCE 4096		
#endif

#ifndef BLOCKSIZE
#define BLOCKSIZE 8
#endif

int * d_auction(int nBidders, int nItems, float * h_benefits);
#endif
