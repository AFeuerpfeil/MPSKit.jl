struct NoExpand <: Algorithm end

function changebonds_left(AL, Cs, alg; kwargs...)
    @info "Old size: $(dim(right_virtualspace(AL)))"
    AL, Cs = changebonds(; expand_rightspace = AL, embed_leftspace = Cs, alg, kwargs...)[[2,end]]
    @info "New size: $(dim(right_virtualspace(AL)))"
    return AL, Cs
end
function changebonds_right(Cs, AR, alg; kwargs...)
    Cs, AR = changebonds(; expand_leftspace = AR, embed_rightspace = Cs, alg, kwargs...)[[1,4]]
    return Cs, AR
end
function changebonds(al,c,ar; kwargs...)
    _, al, c, ar, _ = changebonds(; expand_rightspace = al, expand_leftspace = ar, embed_both = (c,), alg, kwargs...)
    return al, only(c), ar
end

function changebonds(;
        embed_rightspace = missing,
        expand_rightspace = missing,
        expand_leftspace = missing,
        embed_leftspace = missing,
        embed_both = missing,
        alg,
        ac2 = missing,
        expansion_leftspace = ismissing(ac2) ? missing : sup_inf_space(ac2),
        expansion_rightspace = ismissing(expansion_leftspace) ? (ismissing(ac2) ? missing : sup_inf_space(ac2)) : expansion_leftspace,
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
        if space(expand_rightspace) != space(expand_rightspace_new) || true
            if !ismissing(embed_leftspace)
                embed_leftspace = (_embed_left_space(expand_rightspace_new, A, alg) for A in embed_leftspace)
            end
            if !ismissing(embed_both)
                embed_both = (_embed_left_space(expand_rightspace_new, A, alg) for A in embed_both)
            end
        end
        expand_rightspace = expand_rightspace_new
    end
    if !ismissing(expand_leftspace)
        expand_leftspace_new = _expand_rightisometry(expand_leftspace, alg, expansion_rightspace)
        if space(expand_leftspace) != space(expand_leftspace_new) || true
            if !ismissing(embed_rightspace)
                embed_rightspace = (_embed_right_space(A, expand_leftspace_new, alg) for A in embed_rightspace)
            end
            if !ismissing(embed_both)
                embed_both = (_embed_right_space(A, expand_leftspace_new, alg) for A in embed_both)
            end
        end
        expand_leftspace = expand_leftspace_new
    end
    return embed_rightspace, expand_rightspace, embed_both, expand_leftspace, embed_leftspace
end



## Idea of two-site expansion after update: full space in middle is sup(Vl ⊗ p1, Vr ⊗ p2), the compact SVD space is inf(Vl ⊗ p1, Vr ⊗ p2), which consists of kept + truncated space (if one uses a truncated SVD).
## Then, we want to only take samples in the sup ⊖ inf space (previosuly, I took sup - V_kept, which also added states, we just truncated!)
function sup_inf_space(ac2)
    VL = space(ac2, 1) ⊗ space(ac2, 2)
    VR = space(ac2, 3) ⊗ space(ac2, 4)
    return supremum(VL, VR) ⊖ infimum(VL, VR)
end
function _sample_space(space, sup, trscheme)
    sp = ismissing(sup) ? space : infimum(space, sup)
    return sample_space(sp, trscheme)
end
function _expand_leftisometry(A::MPSTensor, alg, expansion_leftspace)
    VL = left_null(A)
    V = _sample_space(right_virtualspace(VL), expansion_leftspace, alg.trscheme)
    XL = randisometry(scalartype(VL), right_virtualspace(VL) ← V)
    return catdomain(A, VL * XL)
end

function _expand_rightisometry(A::MPSTensor, alg, expansion_rightspace)
    return _transpose_front(_expand_rightisometry(_transpose_tail(A; copy = true), alg, expansion_rightspace))
end
function _expand_rightisometry(AR_tail::AbstractTensorMap, alg, expansion_rightspace)
    VR = right_null(AR_tail)
    V = _sample_space(space(VR, 1), expansion_rightspace, alg.trscheme)
    XR = randisometry(scalartype(VR), space(VR, 1) ← V)
    return catcodomain(AR_tail, XR' * VR)
end

function _embed_left_space(A::MPSTensor, C::MPSBondTensor, alg)
    C′ = similar(C, right_virtualspace(A) ← right_virtualspace(C))
    scale!(randn!(C′), alg.noisefactor)
    C′ = TensorKit.absorb!(C′, C)
    return C′
end
function _embed_left_space(A::MPSTensor, Anext::MPSTensor, alg)
    Anext′ = similar(Anext, right_virtualspace(A) ⊗ physicalspace(Anext) ← right_virtualspace(Anext))
    scale!(randn!(Anext′), alg.noisefactor)
    Anext′ = TensorKit.absorb!(Anext′, Anext)
    return Anext′
end

function _embed_right_space(C::MPSBondTensor, A::AbstractTensorMap, alg)
    C′ = similar(C, left_virtualspace(C) ← space(A, 1))
    scale!(randn!(C′), alg.noisefactor)
    C′ = TensorKit.absorb!(C′, C)
    return C′
end
function _embed_right_space(Anext::MPSTensor, A::AbstractTensorMap, alg)
    Anext′ = similar(Anext, left_virtualspace(Anext) ⊗ physicalspace(Anext) ← space(A, 1))
    scale!(randn!(Anext′), alg.noisefactor)
    Anext′ = TensorKit.absorb!(Anext′, Anext)
    return Anext′
end


extract_sector_types(::Type{GradedSpace{S,D}}) where {S<:Sector,D} = (S,)
extract_sector_types(::Type{GradedSpace{ProductSector{T},D}}) where {T<:Tuple,D} = Tuple(T.parameters)
extract_sector_types(sp::GradedSpace) = extract_sector_types(typeof(sp))
function generate_sampling_space(psi::MPSKit.AbstractMPS, cutoff::Integer=100)
    sp = physicalspace(psi.AL[1])
    x = extract_sector_types(sp)
    iterator = Iterators.product((Iterators.take(values(T),cutoff) for T in x)...)
    return typeof(sp)([T=>1 for T in iterator])
end
