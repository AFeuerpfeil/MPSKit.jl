"""
    correlator(ψ, O1, O2, i, j)
    correlator(ψ, O12, i, j)

Compute the 2-point correlator <ψ|O1[i]O2[j]|ψ> for inserting `O1` at `i` and `O2` at `j`.
Also accepts ranges for `j`.

`O1` and `O2` can each be:
- an `MPOTensor` (single-site operator with trivial virtual legs),
- an `AbstractTensorMap{S,N,N}` (nonlocal N-site operator, decomposed internally), or
- a `PeriodicVector{<:MPOTensor}` (site-dependent single-site operators).

    correlator(ψ, O1, O2, O3, O4, i1, i2, i3, i4)

Compute the 4-point correlator <ψ|O1[i1]O2[i2]O3[i3]O4[i4]|ψ> with i1 < i2 < i3 < i4.
Each operator can be an `MPOTensor` or a `PeriodicVector{<:MPOTensor}`.
"""
function correlator end

# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

# Wrap a bare MPOTensor into a PeriodicVector of the given length.
_as_periodic(O::MPOTensor, L::Int) = PeriodicArray(fill(O, L))
_as_periodic(O::PeriodicArray, ::Int) = O

# Advance Vₗ by one site with an intermediate MPO tensor (non-trivial virtual bonds).
# Vₗ has shape [-1 -2; -3] where -2 carries the running MPO virtual bond.
# The output Vₗ has the same shape with updated bonds.
function _transfer_right_mpo(Vₗ, o::MPOTensor, AR)
    return @plansor Vₗ_new[-1 -2; -3] := Vₗ[1 2; 4] * AR[4 5; -3] * o[2 3; 5 -2] *
        conj(AR[1 3; -1])
end

# Push a sequence of MPO tensors (interior pieces, i.e. all but the last) into Vₗ
# using state.AR tensors starting at `start`. Returns (Vₗ_new, next_site).
function _push_ops!(Vₗ, ops, state, start::Int)
    ctr = start
    for o in ops
        Vₗ = _transfer_right_mpo(Vₗ, o, state.AR[ctr])
        ctr += 1
    end
    return Vₗ, ctr
end

# -----------------------------------------------------------------------
# Core implementation: PeriodicVector × PeriodicVector, single-site each
# -----------------------------------------------------------------------

function correlator(
        state::AbstractMPS, O₁s::PeriodicArray{<:MPOTensor, 1},
        O₂s::PeriodicArray{<:MPOTensor, 1}, i::Int, j::Int
    )
    return first(correlator(state, O₁s, O₂s, i, j:j))
end

function correlator(
        state::AbstractMPS, O₁s::PeriodicArray{<:MPOTensor, 1},
        O₂s::PeriodicArray{<:MPOTensor, 1}, i::Int, js::AbstractRange{Int}
    )
    O₁ = O₁s[i]
    first(js) > i || @error "i should be smaller than j ($i, $(first(js)))"
    S₁ = _firstspace(O₁)
    isunitspace(S₁) || throw(ArgumentError("O₁ should start with a trivial leg."))

    G = similar(js, scalartype(state))

    @plansor Vₗ[-1 -2; -3] := state.AC[i][2 3; -3] * removeunit(O₁, 1)[1; 3 -2] *
        conj(state.AC[i][2 1; -1])
    ctr = i + 1

    for (k, j) in enumerate(js)
        O₂ = O₂s[j]
        S₂ = _lastspace(O₂)
        S₂ == S₁' || throw(ArgumentError("O₂ should end with a trivial leg."))
        if j > ctr
            Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
        end
        G[k] = @plansor Vₗ[1 2; 4] * state.AR[j][4 5; 6] * removeunit(O₂, 4)[2 3; 5] *
            conj(state.AR[j][1 3; 6])
        ctr = j
    end
    return G
end

# -----------------------------------------------------------------------
# Existing single MPOTensor interface: dispatches to PeriodicVector core
# -----------------------------------------------------------------------

function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int, j::Int)
    return first(correlator(state, O₁, O₂, i, j:j))
end

function correlator(
        state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int, js::AbstractRange{Int}
    )
    L = length(state)
    return correlator(state, _as_periodic(O₁, L), _as_periodic(O₂, L), i, js)
end

function correlator(
        state::AbstractMPS, O₁₂::AbstractTensorMap{<:Any, S, 2, 2}, i::Int, j
    ) where {S}
    O₁, O₂ = decompose_localmpo(add_util_leg(O₁₂))
    return correlator(state, O₁, O₂, i, j)
end

# -----------------------------------------------------------------------
# Nonlocal 2-point correlator: AbstractTensorMap{S,N,N} with N ≥ 2
#
# i        = start site of O1 (spans N1 sites: i … i+N1-1)
# j/js     = start site of O2 (spans N2 sites: j … j+N2-1), j ≥ i+N1
# -----------------------------------------------------------------------

function correlator(
        state::AbstractMPS,
        O₁::AbstractTensorMap{<:Any, S, N1, N1},
        O₂::AbstractTensorMap{<:Any, S, N2, N2},
        i::Int, j
    ) where {S, N1, N2}
    ops1 = decompose_localmpo(add_util_leg(O₁))
    ops2 = decompose_localmpo(add_util_leg(O₂))
    return _correlator_nonlocal(state, ops1, ops2, i, j)
end

function _correlator_nonlocal(state, ops1, ops2, i::Int, j::Int)
    return first(_correlator_nonlocal(state, ops1, ops2, i, j:j))
end

function _correlator_nonlocal(state, ops1, ops2, i::Int, js::AbstractRange{Int})
    first(js) >= i + length(ops1) ||
        @error "j must be ≥ i + length(O1) ($i + $(length(ops1)), $(first(js)))"

    # Validate boundary legs
    S₁ = _firstspace(ops1[1])
    isunitspace(S₁) || throw(ArgumentError("ops1[1] should start with a trivial leg."))
    S₂ = _lastspace(ops2[end])
    S₂ == S₁' || throw(ArgumentError("ops2[end] should end with a trivial leg."))

    G = similar(js, scalartype(state))

    # Build Vₗ at site i using the first piece of ops1 (removes trivial left leg)
    @plansor Vₗ₀[-1 -2; -3] := state.AC[i][2 3; -3] * removeunit(ops1[1], 1)[1; 3 -2] *
        conj(state.AC[i][2 1; -1])

    # Push remaining interior pieces of O1 (all but last piece which has trivial right leg)
    # If length(ops1) == 1, this is a no-op.
    Vₗ_after_O1, ctr_after_O1 = _push_ops!(Vₗ₀, ops1[2:end], state, i + 1)

    for (k, j) in enumerate(js)
        Vₗ = Vₗ_after_O1
        ctr = ctr_after_O1

        # Free propagation between end of O1 and start of O2
        if j > ctr
            Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
        end

        # Push interior pieces of O2 (all but the last)
        Vₗ_mid, _ = _push_ops!(Vₗ, ops2[1:(end - 1)], state, j)
        last_site = j + length(ops2) - 1

        # Contract last piece of O2 (trivial right leg removed) to get scalar
        O₂_last = ops2[end]
        G[k] = @plansor Vₗ_mid[1 2; 4] * state.AR[last_site][4 5; 6] *
            removeunit(O₂_last, 4)[2 3; 5] * conj(state.AR[last_site][1 3; 6])
    end
    return G
end

# -----------------------------------------------------------------------
# PeriodicVector nonlocal variant:
# O1 spans N1 sites (O1s[i], O1s[i+1], ..., O1s[i+N1-1])
# O2 spans N2 sites (O2s[j], O2s[j+1], ..., O2s[j+N2-1])
# j/js = start site of O2, j ≥ i + N1
# -----------------------------------------------------------------------

function correlator(
        state::AbstractMPS,
        O₁s::PeriodicArray{<:MPOTensor, 1}, N1::Int,
        O₂s::PeriodicArray{<:MPOTensor, 1}, N2::Int,
        i::Int, j
    )
    return _correlator_nonlocal_pv(state, O₁s, N1, O₂s, N2, i, j)
end

# convenience: accept bare MPOTensor for either operator
function correlator(
        state::AbstractMPS,
        O₁, N1::Int,
        O₂, N2::Int,
        i::Int, j
    )
    L = length(state)
    return _correlator_nonlocal_pv(state, _as_periodic(O₁, L), N1,
                                   _as_periodic(O₂, L), N2, i, j)
end

function _correlator_nonlocal_pv(state, O₁s, N1::Int, O₂s, N2::Int, i::Int, j::Int)
    return first(_correlator_nonlocal_pv(state, O₁s, N1, O₂s, N2, i, j:j))
end

function _correlator_nonlocal_pv(
        state, O₁s, N1::Int, O₂s, N2::Int, i::Int, js::AbstractRange{Int}
    )
    first(js) >= i + N1 ||
        @error "j must be ≥ i + N1 ($i + $N1, $(first(js)))"

    O₁_first = O₁s[i]
    S₁ = _firstspace(O₁_first)
    isunitspace(S₁) || throw(ArgumentError("O₁s[i] should start with a trivial leg."))

    G = similar(js, scalartype(state))

    # Build Vₗ at site i from O1s[i]
    @plansor Vₗ₀[-1 -2; -3] := state.AC[i][2 3; -3] * removeunit(O₁_first, 1)[1; 3 -2] *
        conj(state.AC[i][2 1; -1])

    # Push remaining N1-1 pieces of O1
    ops1_rest = [O₁s[i + k] for k in 1:(N1 - 1)]
    Vₗ_after_O1, ctr_after_O1 = _push_ops!(Vₗ₀, ops1_rest, state, i + 1)

    for (idx, j) in enumerate(js)
        Vₗ = Vₗ_after_O1
        ctr = ctr_after_O1

        if j > ctr
            Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
        end

        # Push N2-1 interior pieces of O2
        ops2_head = [O₂s[j + k - 1] for k in 1:(N2 - 1)]
        Vₗ_mid, _ = _push_ops!(Vₗ, ops2_head, state, j)

        last_site = j + N2 - 1
        O₂_last = O₂s[last_site]
        S₂ = _lastspace(O₂_last)
        S₂ == S₁' || throw(ArgumentError("O₂s at last site should end with a trivial leg."))

        G[idx] = @plansor Vₗ_mid[1 2; 4] * state.AR[last_site][4 5; 6] *
            removeunit(O₂_last, 4)[2 3; 5] * conj(state.AR[last_site][1 3; 6])
    end
    return G
end

# -----------------------------------------------------------------------
# 4-point correlator
# i1 < i2 < i3 < i4; O1 starts at i1, O4 ends at i4.
# O2 and O3 are single-site and have trivial virtual legs on both sides.
# -----------------------------------------------------------------------

function correlator(
        state::AbstractMPS,
        O₁, O₂, O₃, O₄,
        i1::Int, i2::Int, i3::Int, i4::Int
    )
    L = length(state)
    O₁s = _as_periodic(O₁, L)
    O₂s = _as_periodic(O₂, L)
    O₃s = _as_periodic(O₃, L)
    O₄s = _as_periodic(O₄, L)
    return _correlator4(state, O₁s, O₂s, O₃s, O₄s, i1, i2, i3, i4)
end

function _correlator4(state, O₁s, O₂s, O₃s, O₄s, i1::Int, i2::Int, i3::Int, i4::Int)
    i1 < i2 < i3 < i4 || @error "Sites must be strictly increasing: ($i1, $i2, $i3, $i4)"

    O₁ = O₁s[i1]
    S₁ = _firstspace(O₁)
    isunitspace(S₁) || throw(ArgumentError("O₁ should start with a trivial leg."))
    S₄ = _lastspace(O₄s[i4])
    S₄ == S₁' || throw(ArgumentError("O₄ should end with a trivial leg."))

    # Build Vₗ at i1
    @plansor Vₗ[-1 -2; -3] := state.AC[i1][2 3; -3] * removeunit(O₁, 1)[1; 3 -2] *
        conj(state.AC[i1][2 1; -1])
    ctr = i1 + 1

    # Propagate to i2 and insert O2
    if i2 > ctr
        Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(i2 - 1)])
    end
    Vₗ = _transfer_right_mpo(Vₗ, O₂s[i2], state.AR[i2])
    ctr = i2 + 1

    # Propagate to i3 and insert O3
    if i3 > ctr
        Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(i3 - 1)])
    end
    Vₗ = _transfer_right_mpo(Vₗ, O₃s[i3], state.AR[i3])
    ctr = i3 + 1

    # Propagate to i4 and close with O4
    if i4 > ctr
        Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(i4 - 1)])
    end
    O₄ = O₄s[i4]
    return @plansor Vₗ[1 2; 4] * state.AR[i4][4 5; 6] * removeunit(O₄, 4)[2 3; 5] *
        conj(state.AR[i4][1 3; 6])
end
