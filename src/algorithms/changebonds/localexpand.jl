struct NoExpand <: Algorithm end

function changebonds_left(AL, C, alg::RandPerturbedExpand)
    AL = _expand_leftisometry(AL, alg)
    C′ = _embed_left_space(AL, C, alg)
    return AL, C′
end
function changebonds_left(AL, C1, C2, alg::RandPerturbedExpand)
    AL = _expand_leftisometry(AL, alg)
    C1′ = _embed_left_space(AL, C1, alg)
    C2′ = _embed_left_space(AL, C2, alg)
    return AL, C1′, C2′
end
function changebonds_right(C, AR, alg::RandPerturbedExpand)
    AR = _expand_rightisometry(AR, alg)
    C′ = _embed_right_space(C, AR, alg)
    return C′, AR
end
function changebonds_right(C1, C2, AR, alg::RandPerturbedExpand)
    AR = _expand_rightisometry(AR, alg)
    C1′ = _embed_right_space(C1, AR, alg)
    C2′ = _embed_right_space(C2, AR, alg)
    return C1′, C2′, AR
end

function changebonds(AL, C, AR, alg::RandPerturbedExpand)
    AL = _expand_leftisometry(AL, alg)
    C = _embed_left_space(AL, C, alg)
    AR = _expand_rightisometry(AR, alg)
    C = _embed_right_space(C, AR, alg)
    return AL, C, AR
end

function _expand_leftisometry(A::MPSTensor, alg)
    VL = left_null(A)
    V = sample_space(right_virtualspace(VL), alg.trscheme)
    XL = randisometry(scalartype(VL), right_virtualspace(VL) ← V)
    A = catdomain(A, VL * XL)
    return A
end
function _expand_rightisometry(A::MPSTensor, alg)
    return _transpose_front(_expand_rightisometry(_transpose_tail(A; copy = true), alg))
end
function _expand_rightisometry(AR_tail::AbstractTensorMap, alg)
    VR = right_null(AR_tail)
    V = sample_space(space(VR, 1), alg.trscheme)
    XR = randisometry(scalartype(VR), space(VR, 1) ← V)
    AR_tail = catcodomain(AR_tail, XR' * VR)
    return AR_tail
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
