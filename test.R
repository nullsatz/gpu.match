#!/usr/local/R-2.15.2/lib64/R/bin/Rscript

args <- commandArgs(trailingOnly=T)
if(length(args) < 1)
	stop('Usage: test.R <test case filename>\n')

inFn <- args[1]
test.case <- as.matrix(read.table(inFn))

dyn.load('/home/bucknerj/gpu.match/match.so')
test.result <- .Call('auction', test.case)
#print(test.result)
