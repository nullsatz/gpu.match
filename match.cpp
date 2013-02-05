// system and standard lib includes
#include<vector>

// R includes
#include<Rcpp.h>
using namespace Rcpp;

// local includes
#include"auction.h"

IntegerMatrix & makeAssi(int nBidders, int nItems, int * assi) {
	IntegerMatrix * rAssi = new IntegerMatrix(nBidders, nItems);
	for(int i = 0; i < nBidders; i++)
		(*rAssi)(i, assi[i]) = 1;
	return *rAssi;
}

// rows: bidders; cols: items
RcppExport SEXP auction(SEXP benefits) {
	NumericMatrix inBenefits(benefits);
	int
		nBidders = inBenefits.nrow(), nItems = inBenefits.ncol();

	std::vector<float> bene(nBidders * nItems);
	for(int i = 0; i < nBidders; i++) {
		int row_i = i * nItems;
		for(int j = 0; j < nItems; j++)
			bene[row_i + j] = inBenefits(i, j); 
	}

	int * assi = d_auction(nBidders, nItems, bene.data());
	IntegerMatrix rAssi = makeAssi(nBidders, nItems, assi);
	return rAssi;
}
