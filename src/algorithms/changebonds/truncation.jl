function noexpand()
    return DynamicTruncation(;maxrank=0,maxrank_max=0,rank_factor=1.0,f=trscheme->RandPerturbedExpand(;trscheme))
end
