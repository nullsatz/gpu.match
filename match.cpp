// system and standard lib includes
#include<vector>

// R includes
#include<Rcpp.h>
using namespace Rcpp;

// local includes
#include"auction.h"

IntegerMatrix & makeAssi(int n, int * assi) {
	IntegerMatrix * rAssi = new IntegerMatrix(n, n);
	for(int i = 0; i < n; i++)
		(*rAssi)(i, assi[i]) = 1;
	return *rAssi;
}

// rows: bidders; cols: items
RcppExport SEXP auction(SEXP benefits) {
	NumericMatrix inBenefits(benefits);
	int n = inBenefits.nrow();

	std::vector<float> bene(n * n);
	for(int i = 0; i < n; i++) {
		int row_i = i * n;
		for(int j = 0; j < n; j++)
			bene[row_i + j] = inBenefits(i, j); 
	}

	int * assi = d_auction(n, bene.data());
	IntegerMatrix rAssi = makeAssi(n, assi);
	return rAssi;
}
