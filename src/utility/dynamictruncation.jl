module DynamicTruncations

const RealOrNothing = Union{Real, Nothing}

import ..MPSKit: Algorithm
using MatrixAlgebraKit
import MatrixAlgebraKit: TruncationStrategy
using DocStringExtensions
using TensorKit: truncspace

export updatetruncation, DynamicTruncation

@doc """
    updatetruncation(alg, iter, ϵ)

Update the truncation tolerance of the algorithm `alg` based on the current iteration `iter` and the current error `ϵ`.
""" updatetruncation
function updatetruncation(alg, args...)
    return alg
end

# Wrapper for dynamic tolerance adjustment
# ----------------------------------------


"""
$(TYPEDEF)

Algorithm wrapper with dynamically adjusted tolerances.

## Fields

$(TYPEDFIELDS)

See also [`updatetruncation`](@ref).
"""
struct DynamicTruncation{F,T1<:RealOrNothing,T2<:RealOrNothing,T3<:RealOrNothing,T4<:RealOrNothing,T5<:RealOrNothing,T6<:RealOrNothing,T7<:RealOrNothing,T8} <: TruncationStrategy
    "parent algorithm"
    f::F # Function of a TruncationStrategy, default just returns the input
    atol::T1
    atol_min::T1 

    rtol::T2
    rtol_min::T2

    maxrank::T3 
    maxrank_max::T3 

    maxerror::T4
    maxerror_min::T4

    filter::T5

    tol_factor::T6
    rank_factor::T7

    space::T8
    function DynamicTruncation(;
            f::F= Base.identity,
            atol=nothing,
            atol_min = isnothing(atol) ? nothing : zero(atol),
            rtol=nothing,
            rtol_min = isnothing(rtol) ? nothing : zero(rtol),
            maxrank=nothing,
            maxrank_max = isnothing(maxrank) ? nothing : typemax(Int),
            maxerror=nothing,
            maxerror_min = isnothing(maxerror) ? nothing : zero(maxerror),
            filter=nothing,
            tol_factor=1.0,
            rank_factor=!isnothing(maxrank) && !isnothing(maxrank_max) ? (maxrank_max/maxrank)^(1/10) : 1.0,
            space = nothing,
        ) where {F}
        @assert isnothing(tol_factor) || 1 >= tol_factor > 0 "tol_factor must be in (0, 1]"
        @assert isnothing(rank_factor) || rank_factor >= 0 "rank_factor must be positive"
        T1 = typeof(atol); T2 = typeof(rtol); T3 = typeof(maxrank); T4 = typeof(maxerror); T5 = typeof(filter); T6 = typeof(tol_factor); T7 = typeof(rank_factor); T8 = typeof(space)
        return new{F,T1,T2,T3,T4,T5,T6,T7,T8}(f, atol, atol_min, rtol, rtol_min, maxrank, maxrank_max, maxerror, maxerror_min, filter, tol_factor, rank_factor, space)
    end
end

function _clamp(a::Nothing,b::Nothing,c::X, factor) where {X}
    return a
end
function _clamp(a::Nothing,b::X,c::Nothing, factor) where {X}
    return a
end
function _clamp(a::Nothing,b::Nothing,c::Nothing, factor)
    return a
end
function _clamp(a::Nothing,b::X,c::Y,factor) where {X, Y}
    return a
end
function _clamp(a,b,c, factor)
    return clamp(a*factor,b,c)
end
function _clamp(a,::Nothing,c::X, factor) where {X}
    return _clamp(a, a, c, factor)
end
function _clamp(a,b,::Nothing, factor)
    return _clamp(a,b,b, factor)
end
function int_clamp(a::Nothing,b,c,factor)
    return a
end
function int_clamp(a,b,c, factor)
    return floor(Int, _clamp(a,b,c,factor))
end

"""
    updatetruncation(alg::DynamicTruncation, iter, ϵ)

Update the truncation tolerance of the algorithm `alg` based on the current iteration `iter` and the current error `ϵ`,
where the new tolerance is given by
"""
function updatetruncation(alg::DynamicTruncation; iter::Integer=0, current_rank::Integer = 0)
    iter = max(iter, one(iter))
    tol_factor = alg.tol_factor^iter 
    rank_factor = alg.rank_factor^iter
    new_atol = _clamp(alg.atol, alg.atol_min, nothing, tol_factor)
    new_rtol = _clamp(alg.rtol, alg.rtol_min, nothing, tol_factor)
    new_maxerror = _clamp(alg.maxerror, alg.maxerror_min, nothing, tol_factor)

    new_maxrank = int_clamp(alg.maxrank, nothing, alg.maxrank_max, rank_factor)
    new_maxrank = isnothing(new_maxrank) ? nothing : max(0, new_maxrank - current_rank)
    if !iszero(current_rank)
        @info "current rank: $current_rank, new maxrank: $new_maxrank"
    end

    strategy = MatrixAlgebraKit.TruncationStrategy(;
        atol = new_atol,
        rtol = new_rtol,
        maxerror = new_maxerror,
        maxrank = new_maxrank,
    )
    if !isnothing(alg.space)
        strategy = (strategy | truncspace(alg.space)) # Guarantees we keep at least the specified space
        if !isnothing(new_maxrank)
            strategy = (strategy & truncrank(new_maxrank)) # Also apply the maxrank constraint after ensuring the space is kept
        end
    end
    
    return alg.f(strategy)
end


end
