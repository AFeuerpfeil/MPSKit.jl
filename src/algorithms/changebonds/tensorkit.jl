
function MatrixAlgebraKit.findtruncated(values::TensorKit.SectorVector, strategy::MatrixAlgebraKit.TruncationUnion)
    inds = map(Base.Fix1(MatrixAlgebraKit.findtruncated, values), strategy.components)
    @assert TensorKit._allequal(keys, inds) "missing blocks are not supported right now"
    sectors = keys(first(inds))
    vals = map(keys(first(inds))) do c
        mapreduce(Base.Fix2(getindex, c), MatrixAlgebraKit._ind_union, inds)
    end
    return TensorKit.SectorDict{eltype(sectors), eltype(vals)}(sectors, vals)
end
function MatrixAlgebraKit.findtruncated_svd(values::TensorKit.SectorVector, strategy::MatrixAlgebraKit.TruncationUnion)
    inds = map(Base.Fix1(MatrixAlgebraKit.findtruncated_svd, values), strategy.components)
    @assert TensorKit._allequal(keys, inds) "missing blocks are not supported right now"
    sectors = keys(first(inds))
    vals = map(keys(first(inds))) do c
        mapreduce(Base.Fix2(getindex, c), MatrixAlgebraKit._ind_union, inds)
    end
    return TensorKit.SectorDict{eltype(sectors), eltype(vals)}(sectors, vals)
end
