struct NoExpand <: Algorithm end

function changebonds_left(AL, C::AbstractTensorMap, alg; kwargs...)
    Anew, Cnew = changebonds_left(AL, (C,), alg; kwargs...)
    return Anew, only(Cnew)
end
function changebonds_left(AL, Cs::Tuple, alg; kwargs...)
    ALnew, Csnew = changebonds(; expand_rightspace = AL, embed_leftspace = Cs, alg, kwargs...)[[2,end]]
    Csnew = collect(Csnew)
    for i in eachindex(Csnew)[1:1]
        @assert ALnew * Csnew[i] ≈ AL * Cs[i] "Error in left embedding is large"
    end
    return ALnew, Csnew
end

function changebonds_right(C::AbstractTensorMap, AR, alg; kwargs...)
    Cnew, ARnew = changebonds_right((C,), AR, alg; kwargs...)
    return only(Cnew), ARnew
end
function changebonds_right(Cs, AR, alg; kwargs...)
    Csnew, ARnew = changebonds(; expand_leftspace = AR, embed_rightspace = Cs, alg, kwargs...)[[1,4]]
    Csnew = collect(Csnew)
    for i in eachindex(Csnew)[1:1]
        @assert Csnew[i] * ARnew ≈ Cs[i] * AR "Error in right embedding is $(norm(Csnew[i] * ARnew - Cs[i] * AR))"
    end
    return Csnew, ARnew
end
function changebonds(al,c,ar, alg; kwargs...)
    _, alnew, cnew, arnew, _ = changebonds(; expand_rightspace = al, expand_leftspace = ar, embed_both = (c,), alg, kwargs...)
    @assert alnew * only(cnew) * arnew ≈ al * c * ar "Error in two-site embedding is $(norm(alnew * only(cnew) * arnew - al * c * ar))"
    return alnew, only(cnew), arnew
end

function changebonds(;
        embed_rightspace = missing,
        expand_rightspace = missing,
        expand_leftspace = missing,
        embed_leftspace = missing,
        embed_both = missing,
        alg,
        ac2 = missing,
        expansion_leftspace =missing, # = ismissing(ac2) ? missing : sup_inf_space(ac2),
        expansion_rightspace = missing #ismissing(expansion_leftspace) ? (ismissing(ac2) ? missing : sup_inf_space(ac2)) : expansion_leftspace,
    )
    return changebonds(
        embed_rightspace, expand_rightspace,embed_both, expand_leftspace, embed_leftspace, alg;
        expansion_leftspace, expansion_rightspace
    )
end
const MissingOrTuple = Union{Missing, <:Tuple}
function changebonds(
        embed_rightspace::MissingOrTuple, expand_rightspace, embed_both::MissingOrTuple, expand_leftspace, embed_leftspace::MissingOrTuple, alg;
        expansion_leftspace, expansion_rightspace
    )
    if !ismissing(expand_rightspace)
        expand_rightspace_new = _expand_leftisometry(expand_rightspace, alg, expansion_leftspace)
        if space(expand_rightspace) != space(expand_rightspace_new)
            println("happened1")
            if !ismissing(embed_leftspace)
                embed_leftspace = collect(_embed_left_space(expand_rightspace_new, expand_rightspace,A) for A in embed_leftspace)
            end
            if !ismissing(embed_both)
                embed_both = collect(_embed_left_space(expand_rightspace_new, expand_rightspace, A) for A in embed_both)
            end
        else 
            @assert expand_rightspace ≈ expand_rightspace_new "Error: expand_rightspace is not approximately equal to expand_rightspace_new, but their spaces are the same. This should not happen."
        end
        expand_rightspace = expand_rightspace_new
    end
    if !ismissing(expand_leftspace)
        expand_leftspace_new = _expand_rightisometry(expand_leftspace, alg, expansion_rightspace)
        if space(expand_leftspace) != space(expand_leftspace_new)
            println("happened2")
            if !ismissing(embed_rightspace)
                embed_rightspace = collect(_embed_right_space(A,expand_leftspace, expand_leftspace_new) for A in embed_rightspace)
            end
            if !ismissing(embed_both)
                embed_both = collect(_embed_right_space(A, expand_leftspace, expand_leftspace_new) for A in embed_both)
            end
        end
        expand_leftspace = expand_leftspace_new
    end
    return embed_rightspace, expand_rightspace, embed_both, expand_leftspace, embed_leftspace
end



## Idea of two-site expansion after update: full space in middle is sup(Vl ⊗ p1, Vr ⊗ p2), the compact SVD space is inf(Vl ⊗ p1, Vr ⊗ p2), which consists of kept + truncated space (if one uses a truncated SVD).
## Then, we want to only take samples in the sup ⊖ inf space (previosuly, I took sup - V_kept, which also added states, we just truncated!)
function sup_inf_space(ac2)
    VL = fuse(space(ac2, 1) ⊗ space(ac2, 2))
    VR = fuse(space(ac2, 3) ⊗ space(ac2, 4))
    @show VL, VR 
    @show supremum(VL, VR), infimum(VL, VR)
    @show supremum(VL, VR) ⊖ infimum(VL, VR)
    return supremum(VL, VR) ⊖ infimum(VL, VR)
end
function _sample_space(space, sup, trscheme)
    sp = ismissing(sup) ? space : infimum(space, sup)
    return sample_space(sp, trscheme)
end
function _expand_leftisometry(A::MPSTensor, alg, expansion_leftspace)
    VL = left_null(A)
    V = _sample_space(right_virtualspace(VL), expansion_leftspace, alg.trscheme)
    # @show right_virtualspace(VL), expansion_leftspace, V
    dim(V) == 0 && return A
    XL = randisometry(scalartype(VL), right_virtualspace(VL) ← V)
    x = catdomain(A, VL * XL)
    @tensor I1[right'; right] := x[phys,left,right] * conj(x[phys,left,right'])
    @assert I1 ≈ one(I1) "I1 is $I1"
    return x
end

function _expand_rightisometry(A::MPSTensor, alg, expansion_rightspace)
    return _transpose_front(_expand_rightisometry(_transpose_tail(A; copy = true), alg, expansion_rightspace))
end
function _expand_rightisometry(AR_tail::AbstractTensorMap, alg, expansion_rightspace)
    VR = right_null(AR_tail)
    @tensor overlap[left'; left] := AR_tail[left,phys,right] * conj(VR[left',phys,right])
    @assert abs(norm(overlap))<1e-10 "Error in overlap is $(norm(overlap))"

    V = _sample_space(space(VR, 1), expansion_rightspace, alg.trscheme)
    dim(V) == 0 && return AR_tail
    XR = randisometry(scalartype(VR), space(VR, 1) ← V)
    b = XR' * VR
    @tensor I1[left; left'] := b[left,phys,right] * conj(b[left',phys,right])
    @assert I1 ≈ one(I1) "Error in b is $(norm(I1 - one(I1))) $(norm(I1)), $(norm(one(I1)))"
    
    x = catcodomain(AR_tail, b)
    x = _transpose_front(x)
    @tensor I1[left'; left] := x[left,phys,right] * conj(x[left',phys,right])
    @assert I1 ≈ one(I1) "Error is $(norm(I1 - one(I1))) $(norm(I1)), $(norm(one(I1)))"
    return _transpose_tail(x)
end

function _embed_left_space(A::MPSTensor, C::MPSBondTensor, alg::Algorithm)
    C′ = similar(C, right_virtualspace(A) ← right_virtualspace(C))
    scale!(randn!(C′), alg.noisefactor)
    C′ = TensorKit.absorb!(C′, C)
    return C′
end
function _embed_left_space(A::MPSTensor, Anext::MPSTensor, alg::Algorithm)
    Anext′ = similar(Anext, right_virtualspace(A) ⊗ physicalspace(Anext) ← right_virtualspace(Anext))
    scale!(randn!(Anext′), alg.noisefactor)
    Anext′ = TensorKit.absorb!(Anext′, Anext)
    return Anext′
end

function _embed_right_space(C::MPSBondTensor, A::AbstractTensorMap, alg::Algorithm)
    C′ = similar(C, left_virtualspace(C) ← space(A, 1))
    scale!(randn!(C′), alg.noisefactor)
    C′ = TensorKit.absorb!(C′, C)
    return C′
end
function _embed_right_space(Anext::MPSTensor, A::AbstractTensorMap, alg::Algorithm)
    Anext′ = similar(Anext, left_virtualspace(Anext) ⊗ physicalspace(Anext) ← space(A, 1))
    scale!(randn!(Anext′), alg.noisefactor)
    Anext′ = TensorKit.absorb!(Anext′, Anext)
    return Anext′
end

function _embed_left_space(A::MPSTensor, Aold::MPSTensor, C::MPSBondTensor)
    @tensor C_new[-1; -2] := conj(A[1,2;-1])*Aold[1,2;3] *C[3; -2]
    return add_noise(C_new)
end
function _embed_left_space(A::MPSTensor, Aold::MPSTensor, C::MPSTensor)
    @tensor C_new[-1,-2; -3] := conj(A[1,2;-1])*Aold[1,2;3]*C[3,-2; -3]
    return add_noise(C_new)
end
function _embed_right_space(C::MPSBondTensor, Aold::AbstractTensorMap, A::AbstractTensorMap)
    @tensor C_new[-1; -2] := C[-1; 1]*Aold[1,2;3]*conj(A[-2,2; 3])
    return add_noise(C_new)
end
function _embed_right_space(C::MPSTensor, Aold::AbstractTensorMap, A::AbstractTensorMap)
    @tensor C_new[-1,-2; -3] := C[-1,-2; 1]*Aold[1,2;3]*conj(A[-3,2; 3])
    return add_noise(C_new)
end

function add_noise(A::AbstractTensorMap, noisefactor=eps()^(3/4))
    A′ = similar(A)
    scale!(randn!(A′), noisefactor)
    return A + A′
end





extract_sector_types(::Type{GradedSpace{S,D}}) where {S<:Sector,D} = (S,)
extract_sector_types(::Type{GradedSpace{ProductSector{T},D}}) where {T<:Tuple,D} = Tuple(T.parameters)
extract_sector_types(sp::GradedSpace) = extract_sector_types(typeof(sp))
function generate_sampling_space(psi::MPSKit.AbstractMPS; cutoff::Integer=50, minsize::Integer=1)
    sp = physicalspace(psi.AL[1])
    I = sectortype(sp)
    types = extract_sector_types(sp)
    iterators = [values(T) for T in types]
    iterator = constrained_product(iterators, cutoff)

    r = collect(I(T) => minsize for T in iterator)
    if sp isa GradedSpace
        b=TensorKit.SectorDict{I, Int}(r)
        return GradedSpace{I, TensorKit.SectorDict{I, Int}}(b, false)
    end
    return typeof(sp)(r)
end

function constrained_product(iters, cutoff)
    N = length(iters)

    # Materialize prefixes up to cutoff (0..cutoff -> cutoff + 1 elements)
    prefixes = [collect(Iterators.take(it, cutoff + 1)) for it in iters]
    cutoff_sq = cutoff^2

    # Generate tuples whose index-vector lies inside an L2 ball (sphere)
    # i.e. sum(i.^2) <= cutoff^2. We prune by limiting each coordinate
    # to floor(sqrt(remaining_budget_sq)) and to the available prefix length.
    return Channel() do channel
        function recurse(dim, current_sqsum, current_tuple)
            # available maximum index for this dimension from collected prefix
            max_avail = length(prefixes[dim]) - 1
            if max_avail < 0
                return
            end

            if dim == N
                rem_sq = cutoff_sq - current_sqsum
                max_i = min(max_avail, floor(Int, sqrt(max(0, rem_sq))))
                if max_i >= 0
                    for i in 0:max_i
                        put!(channel, (current_tuple..., prefixes[dim][i+1]))
                    end
                end
            else
                rem_sq = cutoff_sq - current_sqsum
                max_i = min(max_avail, cutoff, floor(Int, sqrt(max(0, rem_sq))))
                if max_i >= 0
                    for i in 0:max_i
                        recurse(dim + 1, current_sqsum + i*i, (current_tuple..., prefixes[dim][i+1]))
                    end
                end
            end
        end

        recurse(1, 0, ())
    end
end
