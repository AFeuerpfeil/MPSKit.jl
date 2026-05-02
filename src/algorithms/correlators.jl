"""
    correlator(ψ, (O1, O2, ..., ON), (is, ds1, ..., dsN-1); connected=false)

Compute the N-point correlator `<ψ|O1[i] O2[i+d1] O3[i+d1+d2] ... ON[i+d1+…+dN-1]|ψ>`
for all combinations of starting positions and distances.

Index tuple `(is, ds1, …, dsN-1)`:
- `is`   : positions of O1 (Integer or AbstractRange{Int})
- `dsk`  : distance from operator k to operator k+1, k = 1…N-1

All distances must be ≥ 1. The output is an array of shape
`(length(is), length(ds1), …, length(dsN-1))`, with singleton dimensions squeezed out
when the corresponding index was given as a plain Integer.

Each operator can be:
- an `MPOTensor` (single-site with trivial virtual legs),
- a `PeriodicArray{<:MPOTensor,1}` (site-dependent single-site operators), or
- an `AbstractTensorMap{S,K,K}` with K ≥ 3 (multi-site, decomposed internally).

If `connected=true`, subtract the disconnected contribution via the full cumulant expansion
(inclusion-exclusion over all proper set partitions).

Convenience dispatchers for small N:

    correlator(ψ, O, i; connected=false)
    correlator(ψ, O1, O2, i, d; connected=false)
    correlator(ψ, O1, O2, O3, i, d1, d2; connected=false)
    correlator(ψ, O1, O2, O3, O4, i, d1, d2, d3; connected=false)
"""
function correlator end

# -----------------------------------------------------------------------
# Small-N convenience dispatchers
# -----------------------------------------------------------------------

function correlator(state::AbstractMPS, O, i; kwargs...)
    return correlator(state, (O,), (i,); kwargs...)
end

function correlator(state::AbstractMPS, O1, O2, i, d; kwargs...)
    return correlator(state, (O1, O2), (i, d); kwargs...)
end

function correlator(state::AbstractMPS, O1, O2, O3, i, d1, d2; kwargs...)
    return correlator(state, (O1, O2, O3), (i, d1, d2); kwargs...)
end

function correlator(
        state::AbstractMPS, O1, O2, O3, O4, i, d1, d2, d3; kwargs...
    )
    return correlator(state, (O1, O2, O3, O4), (i, d1, d2, d3); kwargs...)
end

# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

_as_periodic(O, L::Int) = PeriodicArray(fill(O, L))
_as_periodic(O::PeriodicArray, ::Int) = O

function _transfer_right_mpo(Vₗ, o::MPOTensor, AR)
    return @plansor Vₗ_new[-1 -2; -3] := Vₗ[1 2; 4] * AR[4 5; -3] * o[2 3; 5 -2] *
        conj(AR[1 3; -1])
end

function _expval_mpotensor(state::AbstractMPS, O::MPOTensor, site::Int)
    return local_expectation_value1(state, site, removeunit(removeunit(O, 1), 3))
end

function _push_ops!(Vₗ, ops, state, start::Int)
    ctr = start
    for o in ops
        Vₗ = _transfer_right_mpo(Vₗ, o, state.AR[ctr])
        ctr += 1
    end
    return Vₗ, ctr
end

# Normalize to PeriodicArray{<:MPOTensor,1} (single-site) or Vector{<:MPOTensor} (multi-site).
_decompose_localmpo(O::AbstractTensorMap{<:Any, S, N, N}) where {S, N} = decompose_localmpo(add_util_leg(O))
_decompose_localmpo(O::AbstractVector) = O

# All MPOTensor pieces for operator op starting at site s.
_op_pieces(op::PeriodicArray{<:MPOTensor, 1}, s::Int) = [op[s]]
_op_pieces(op::PeriodicArray, s::Int) = op[s]
_op_pieces(op::Vector, ::Int) = op

# -----------------------------------------------------------------------
# Set partition utilities for connected (cumulant) correlators
# -----------------------------------------------------------------------

function _set_partitions(elements::Vector)
    isempty(elements) && return [Vector{eltype(elements)}[]]
    first_elem = elements[1]
    rest_parts = _set_partitions(elements[2:end])
    result = []
    for partition in rest_parts
        push!(result, [[first_elem]; partition])
        for (i, _) in enumerate(partition)
            new_part = copy.(partition)
            push!(new_part[i], first_elem)
            push!(result, new_part)
        end
    end
    return result
end

# All set partitions of 1:N except the trivial full-set partition.
function _proper_set_partitions(N::Int)
    return filter(p -> length(p) > 1, _set_partitions(collect(1:N)))
end

# Möbius sign: (-1)^(|π|+1) * (|π|-1)!
_partition_sign(π) = (-1)^(length(π) + 1) * factorial(length(π) - 1)

# -----------------------------------------------------------------------
# Tuple interface: main entry point
# -----------------------------------------------------------------------

# `ops`  : NTuple of operators
# `idxs` : NTuple of N entries — first is positions of op 1, rest are distances
function correlator(state::AbstractMPS, ops::Tuple, idxs::Tuple; connected::Bool = false)
    N = length(ops)
    N == length(idxs) ||
        throw(ArgumentError("ops and idxs must have the same length (got $N vs $(length(idxs)))"))
    L = length(state)
    ops_norm = map(o -> _decompose_localmpo.(_as_periodic(o, L)), ops)
    # Track which dimensions were scalar (Integer) so we can squeeze them at the end.
    scalar_mask = map(idx -> idx isa Integer, idxs)
    idx_ranges  = map(collect, idxs)
    G = _correlatorN(state, ops_norm, idx_ranges, connected)
    # Squeeze scalar dimensions (drop size-1 dims that came from plain Int indices).
    return _squeeze(G, scalar_mask)
end

# Drop dimensions where scalar_mask is true.
function _squeeze(G::AbstractArray, mask)
    dims_to_drop = Tuple(findall(identity, mask))
    isempty(dims_to_drop) && return G
    return dropdims(G; dims = dims_to_drop)
end

# -----------------------------------------------------------------------
# Core N-point engine
# -----------------------------------------------------------------------

# Output: Array of shape (length(is), length(ds1), …, length(dsN-1)).
# idx_ranges = (is, ds1, …, dsN-1), each an AbstractRange{Int}.
function _correlatorN(state::AbstractMPS, ops_pv, idx_ranges, connected::Bool)
    N  = length(ops_pv)
    is = idx_ranges[1]

    out_size = ntuple(k -> length(idx_ranges[k]), N)
    G = zeros(scalartype(state), out_size...)

    # Pre-compute partitions once if needed.
    partitions = connected ? _proper_set_partitions(N) : nothing

    for (ii, i) in enumerate(is)
        pieces1 = _op_pieces(ops_pv[1], i)
        O1_head = pieces1[1]
        S₁ = _firstspace(O1_head)
        isunitspace(S₁) ||
            throw(ArgumentError("First operator must have a trivial left virtual leg."))

        @plansor Vₗ[-1 -2; -3] := state.AC[i][2 3; -3] * removeunit(O1_head, 1)[1; 3 -2] *
            conj(state.AC[i][2 1; -1])
        Vₗ, ctr = _push_ops!(Vₗ, pieces1[2:end], state, i + 1)

        _fill_G!(G, (ii,), state, ops_pv, idx_ranges, S₁, Vₗ, ctr, 2,
                 connected, partitions)
    end

    return G
end

# Recursively fill G by iterating over distance dimensions depth=2..N.
# idx_prefix : tuple of already-fixed output indices (length = depth-1)
# Vₗ         : boundary vector after applying ops 1..depth-1
# ctr        : next site (one past the last site of op depth-1)
function _fill_G!(
        G, idx_prefix, state, ops_pv, idx_ranges, S₁, Vₗ, ctr, depth,
        connected, partitions
    )
    N    = length(ops_pv)
    ds_k = idx_ranges[depth]

    for (di, d) in enumerate(ds_k)
        d >= 1 || throw(ArgumentError("All distances must be ≥ 1 (got $d at operator $depth)"))
        site_k = ctr + d - 1

        # Propagate across the gap.
        Vₗ_k = site_k > ctr ? Vₗ * TransferMatrix(state.AR[ctr:(site_k - 1)]) : Vₗ

        pieces_k = _op_pieces(ops_pv[depth], site_k)
        new_idx  = (idx_prefix..., di)

        if depth == N
            # Last operator: close to a scalar.
            last_site = site_k + length(pieces_k) - 1
            Vₗ_mid, _ = _push_ops!(Vₗ_k, pieces_k[1:(end - 1)], state, site_k)
            ON_last = pieces_k[end]
            S_last  = _lastspace(ON_last)
            S_last == S₁' ||
                throw(ArgumentError("Last operator must end with a trivial right virtual leg."))
            val = @plansor Vₗ_mid[1 2; 4] * state.AR[last_site][4 5; 6] *
                removeunit(ON_last, 4)[2 3; 5] * conj(state.AR[last_site][1 3; 6])
            G[new_idx...] = val

            # Subtract disconnected contributions (cumulant expansion).
            if connected
                sites_here = _recover_sites(idx_ranges, ops_pv, new_idx)
                for partition in partitions
                    sgn = _partition_sign(partition)
                    contrib = prod(partition) do block
                        sub_ops   = tuple(ops_pv[block]...)
                        sub_sites = tuple(sites_here[block]...)
                        _correlator_absolute(state, sub_ops, sub_sites)
                    end
                    G[new_idx...] -= sgn * contrib
                end
            end
        else
            # Intermediate operator: insert and recurse.
            Vₗ_next = _transfer_right_mpo(Vₗ_k, pieces_k[1], state.AR[site_k])
            Vₗ_next, ctr_next = _push_ops!(Vₗ_next, pieces_k[2:end], state, site_k + 1)
            _fill_G!(G, new_idx, state, ops_pv, idx_ranges, S₁, Vₗ_next, ctr_next,
                     depth + 1, connected, partitions)
        end
    end
end

# N=1: no recursion needed, return expectation values directly.
function _correlatorN(
        state::AbstractMPS, ops_pv::Tuple{Any}, idx_ranges::Tuple{AbstractRange{Int}},
        ::Bool
    )
    is = idx_ranges[1]
    return [_expval_mpotensor(state, ops_pv[1][i], i) for i in is]
end

# -----------------------------------------------------------------------
# Helpers for connected correction
# -----------------------------------------------------------------------

# Recover absolute site positions from the output-array multi-index.
# idx_ranges[1] = positions of op 1; idx_ranges[k] = distances from end of op k-1 (k ≥ 2).
function _recover_sites(idx_ranges, ops_pv, multi_idx)
    N = length(idx_ranges)
    sites = Vector{Int}(undef, N)
    sites[1] = idx_ranges[1][multi_idx[1]]
    for k in 2:N
        width_prev = length(_op_pieces(ops_pv[k - 1], sites[k - 1]))
        sites[k] = sites[k - 1] + width_prev + idx_ranges[k][multi_idx[k]] - 1
    end
    return sites
end

# Compute a correlator for a specific set of absolute site positions (no ranges).
# Used by the cumulant expansion for sub-correlators in each partition block.
function _correlator_absolute(state::AbstractMPS, ops_pv::Tuple, sites::Tuple)
    N = length(ops_pv)
    N == 1 && return _expval_mpotensor(state, ops_pv[1][sites[1]], sites[1])

    pieces1 = _op_pieces(ops_pv[1], sites[1])
    S₁ = _firstspace(pieces1[1])
    @plansor Vₗ[-1 -2; -3] := state.AC[sites[1]][2 3; -3] *
        removeunit(pieces1[1], 1)[1; 3 -2] * conj(state.AC[sites[1]][2 1; -1])
    Vₗ, ctr = _push_ops!(Vₗ, pieces1[2:end], state, sites[1] + 1)

    for k in 2:(N - 1)
        site_k = sites[k]
        Vₗ = site_k > ctr ? Vₗ * TransferMatrix(state.AR[ctr:(site_k - 1)]) : Vₗ
        pieces_k = _op_pieces(ops_pv[k], site_k)
        Vₗ = _transfer_right_mpo(Vₗ, pieces_k[1], state.AR[site_k])
        Vₗ, ctr = _push_ops!(Vₗ, pieces_k[2:end], state, site_k + 1)
    end

    site_N = sites[N]
    Vₗ = site_N > ctr ? Vₗ * TransferMatrix(state.AR[ctr:(site_N - 1)]) : Vₗ
    piecesN   = _op_pieces(ops_pv[N], site_N)
    last_site = site_N + length(piecesN) - 1
    Vₗ_mid, _ = _push_ops!(Vₗ, piecesN[1:(end - 1)], state, site_N)
    ON_last = piecesN[end]
    return @plansor Vₗ_mid[1 2; 4] * state.AR[last_site][4 5; 6] *
        removeunit(ON_last, 4)[2 3; 5] * conj(state.AR[last_site][1 3; 6])
end
