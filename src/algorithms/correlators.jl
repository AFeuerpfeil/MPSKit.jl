"""
    correlator(ψ, (O1, O2, ..., ON), (is, ds1, ..., dsM-1); connected=false)

Compute the N-point correlator `<ψ|O1[i] O2[...] ... ON[...]|ψ>` for all combinations
of starting positions and distances.

Each operator is first decomposed into K pieces via `_decompose_localmpo`. The total number
of `idxs` entries must equal the total number of pieces across all operators:
`length(idxs) == sum(num_pieces(op) for op in ops)`.

Index tuple `(is, ds1, …, dsM-1)` where M = total pieces:
- `is`   : positions of first piece of O1 (Integer or AbstractRange{Int})
- `dsk`  : distance from end of piece k to start of piece k+1, k = 1…M-1

All distances must be ≥ 1. MPOTensors (with virtual legs) count as 1 piece;
raw multi-site TensorMaps (like `S_z_S_z()`) are decomposed into K pieces where K equals
the number of physical sites spanned.

The output is an array of shape `(length(is), length(ds1), …, length(dsM-1))`, with
singleton dimensions squeezed out when the corresponding index was given as a plain Integer.

Each operator can be:
- an `MPOTensor` (single-site with virtual legs, counts as 1 piece),
- a `PeriodicArray{<:MPOTensor,1}` (site-dependent single-site operators), or
- an `AbstractTensorMap{S,K,K}` with K ≥ 3 (multi-site, decomposed into K-1 pieces).

If `connected=true`, subtract the disconnected contribution via the full cumulant expansion
(inclusion-exclusion over all proper set partitions at the operator level).

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
function correlator(state::AbstractMPS, args...; kwargs...)
    indices = findall(arg -> arg isa Integer || arg isa AbstractRange, args)
    ops = setdiff(args, args[indices])
    # return state, tuple(ops...), tuple(args[indices]...)
    return correlator(state, tuple(ops...), tuple(args[indices]...); kwargs...)
end

# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

_as_periodic(O, L::Int) = PeriodicArray(fill(O, L))
_as_periodic(O::PeriodicArray, ::Int) = O

function _transfer_right_mpo(Vₗ, o::MPOTensor, AR, backend = DefaultBackend(), allocator = BufferAllocator())
    return @plansor backend=backend allocator=allocator Vₗ_new[-1 -2; -3] := Vₗ[1 2; 4] * AR[4 5; -3] * o[2 3; 5 -2] *
        conj(AR[1 3; -1])
end

function _expval_mpotensor(state::AbstractMPS, O::MPOTensor, site::Int, backend = DefaultBackend(), allocator = BufferAllocator())
    return local_expectation_value1(state, site, removeunit(removeunit(O, 1), 3))
end

function _push_ops!(Vₗ, ops, state, start::Int, backend = DefaultBackend(), allocator = BufferAllocator())
    ctr = start
    for o in ops
        Vₗ = _transfer_right_mpo(Vₗ, o, state.AR[ctr], backend, allocator)
        ctr += 1
    end
    return Vₗ, ctr
end

# Normalize to PeriodicArray{<:MPOTensor,1} (single-site) or Vector{<:MPOTensor} (multi-site).
# MPOTensors (already have virtual legs) are kept as a 1-element vector — no SVD decomposition.
_decompose_localmpo(O::AbstractVector) = _decompose_localmpo.(O)
_decompose_localmpo(O) = decompose_localmpo(add_util_leg(O))

# All MPOTensor pieces for operator op starting at site s.
_op_pieces(op::PeriodicArray{<:MPOTensor, 1}, s::Int) = [op[s]]
_op_pieces(op::PeriodicArray, s::Int) = op[s]
_op_pieces(op::Vector, ::Int) = op

# Width = number of pieces a normalized op produces.
_op_width(op::PeriodicArray{<:MPOTensor, 1}) = 1
_op_width(op::PeriodicArray) = length(op[1])

# Expand normalized op into one PeriodicArray{<:MPOTensor,1} per piece position.
_flatten_op(op::PeriodicArray{<:MPOTensor, 1}) = [op]
function _flatten_op(op::PeriodicArray)
    L = length(op.data)
    w = length(op[1])
    return [PeriodicArray([op[s][j] for s in 1:L]) for j in 1:w]
end

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
# `idxs` : entries — first is positions of first piece, rest are distances between consecutive pieces
# Total length(idxs) == sum of piece counts across all operators.
function correlator(
        state::AbstractMPS, ops::Tuple, idxs::Tuple;
        connected::Bool = false, scheduler = Defaults.scheduler[],
        backend::AbstractBackend = DefaultBackend(), allocator = BufferAllocator()
    )
    L = length(state)
    ops_norm = map(o -> _decompose_localmpo.(_as_periodic(o, L)), ops)
    widths  = map(_op_width, ops_norm)
    K_total = sum(widths)
    K_total == length(idxs) ||
        throw(ArgumentError(
            "Total operator pieces ($K_total) must equal length(idxs) " *
            "($(length(idxs))). Operator widths: $(Tuple(widths))."
        ))

    # Flatten: each piece of each operator becomes a separate entry.
    flat_ops = Tuple(Iterators.flatten(_flatten_op(op) for op in ops_norm))
    # op_ends[k] = flat index of last piece of original operator k.
    op_ends = Tuple(cumsum(widths))

    # Track which dimensions were scalar (Integer) so we can squeeze them at the end.
    scalar_mask = map(idx -> idx isa Integer, idxs)
    idx_ranges  = map(collect, idxs)
    G = _correlatorN(state, flat_ops, idx_ranges, connected, ops_norm, op_ends; scheduler, backend, allocator)
    # Squeeze scalar dimensions (drop size-1 dims that came from plain Int indices).
    return _squeeze(G, scalar_mask)
end

# Drop dimensions where scalar_mask is true.
function _squeeze(G::AbstractArray, mask)
    dims_to_drop = Tuple(findall(identity, mask))
    isempty(dims_to_drop) && return G
    G_ = dropdims(G; dims = dims_to_drop)
    ndims(G_) == 0 && return G_[]
    return G_
end

# -----------------------------------------------------------------------
# Core N-point engine
# -----------------------------------------------------------------------

# Output: Array of shape (length(is), length(ds1), …, length(dsK-1)) where K = total pieces.
# idx_ranges = (is, ds1, …, dsK-1), each an AbstractRange{Int}.
# ops_norm and op_ends are needed for the connected cumulant correction (operator-level partitions).
function _correlatorN(
        state::AbstractMPS, flat_ops_pv, idx_ranges, connected::Bool, ops_norm, op_ends;
        scheduler = Defaults.scheduler[], backend = DefaultBackend(), allocator = BufferAllocator()
    )
    N  = length(flat_ops_pv)
    is = idx_ranges[1]

    out_size = ntuple(k -> length(idx_ranges[k]), N)
    G = zeros(scalartype(state), out_size...)

    # Partitions are over original operators, not flat pieces.
    N_ops = length(op_ends)
    partitions = connected ? _proper_set_partitions(N_ops) : nothing

    # @tasks 
    for (ii, i) in collect(enumerate(is))
        # @set scheduler = scheduler
        pieces1 = _op_pieces(flat_ops_pv[1], i)
        O1_head = pieces1[1]
        S₁ = _firstspace(O1_head)
        isunitspace(S₁) ||
            throw(ArgumentError("First operator must have a trivial left virtual leg."))

        @plansor backend=backend allocator=allocator Vₗ[-1 -2; -3] := state.AC[i][2 3; -3] * removeunit(O1_head, 1)[1; 3 -2] *
            conj(state.AC[i][2 1; -1])
        Vₗ, ctr = _push_ops!(Vₗ, pieces1[2:end], state, i + 1, backend, allocator)

        _fill_G!(G, (ii,), state, flat_ops_pv, idx_ranges, S₁, Vₗ, ctr, 2,
                 connected, partitions, ops_norm, op_ends, backend, allocator)
    end

    return G
end

# Recursively fill G by iterating over distance dimensions depth=2..N (flat piece count).
# idx_prefix : tuple of already-fixed output indices (length = depth-1)
# Vₗ         : boundary vector after applying flat ops 1..depth-1
# ctr        : next site (one past the last site of flat op depth-1)
function _fill_G!(
        G, idx_prefix, state, flat_ops_pv, idx_ranges, S₁, Vₗ, ctr, depth,
        connected, partitions, ops_norm, op_ends, backend, allocator
    )
    N    = length(flat_ops_pv)
    ds_k = idx_ranges[depth]

    Vₗ_prev = Vₗ
    ctr_prev = ctr

    for (di, d) in enumerate(ds_k)
        d >= 1 || throw(ArgumentError("All distances must be ≥ 1 (got $d at piece $depth)"))
        site_k = ctr + d - 1

        # Propagate across the incremental gap from the previous site.
        Vₗ_k = site_k > ctr_prev ? Vₗ_prev * TransferMatrix(state.AR[ctr_prev:(site_k - 1)], backend, allocator) : Vₗ_prev
        Vₗ_prev = Vₗ_k
        ctr_prev = site_k

        pieces_k = _op_pieces(flat_ops_pv[depth], site_k)
        new_idx  = (idx_prefix..., di)

        if depth == N
            # Last piece: close to a scalar.
            last_site = site_k + length(pieces_k) - 1
            Vₗ_mid, _ = _push_ops!(Vₗ_k, pieces_k[1:(end - 1)], state, site_k, backend, allocator)
            ON_last = pieces_k[end]
            S_last  = _lastspace(ON_last)
            S_last == S₁' ||
                throw(ArgumentError("Last operator must end with a trivial right virtual leg."))
            val = @plansor backend=backend allocator=allocator Vₗ_mid[1 2; 4] * state.AR[last_site][4 5; 6] *
                removeunit(ON_last, 4)[2 3; 5] * conj(state.AR[last_site][1 3; 6])
            G[new_idx...] = val

            # Subtract disconnected contributions (cumulant expansion at operator level).
            if connected
                flat_sites = _recover_sites_flat(idx_ranges, new_idx)
                N_ops = length(op_ends)
                op_all_sites = Vector{Vector{Int}}(undef, N_ops)
                prev = 0
                for k in 1:N_ops
                    op_all_sites[k] = flat_sites[(prev + 1):op_ends[k]]
                    prev = op_ends[k]
                end
                for partition in partitions
                    sgn = _partition_sign(partition)
                    contrib = prod(partition) do block
                        sub_ops   = tuple(ops_norm[block]...)
                        sub_sites = tuple(op_all_sites[block]...)
                        _correlator_absolute(state, sub_ops, sub_sites, backend, allocator)
                    end
                    G[new_idx...] -= sgn * contrib
                end
            end
        else
            # Intermediate piece: insert and recurse.
            Vₗ_next = _transfer_right_mpo(Vₗ_k, pieces_k[1], state.AR[site_k], backend, allocator)
            Vₗ_next, ctr_next = _push_ops!(Vₗ_next, pieces_k[2:end], state, site_k + 1, backend, allocator)
            _fill_G!(G, new_idx, state, flat_ops_pv, idx_ranges, S₁, Vₗ_next, ctr_next,
                     depth + 1, connected, partitions, ops_norm, op_ends, backend, allocator)
        end
    end
end

# N=1: no recursion needed, return expectation values directly.
function _correlatorN(
        state::AbstractMPS, flat_ops_pv::Tuple{Any}, idx_ranges::Tuple{AbstractRange{Int}},
        ::Bool, ops_norm, op_ends;
        backend = DefaultBackend(), allocator = BufferAllocator(), scheduler...
    )
    is = idx_ranges[1]
    return [_expval_mpotensor(state, flat_ops_pv[1][i], i, backend, allocator) for i in is]
end

# -----------------------------------------------------------------------
# Helpers for connected correction
# -----------------------------------------------------------------------

# Recover absolute site positions for each flat piece.
# idx_ranges[1] = starting positions of piece 1; idx_ranges[k] = distance from end of piece k-1.
# Each flat piece has width 1, so site[k] = site[k-1] + d[k].
function _recover_sites_flat(idx_ranges, multi_idx)
    N = length(idx_ranges)
    sites = Vector{Int}(undef, N)
    sites[1] = idx_ranges[1][multi_idx[1]]
    for k in 2:N
        sites[k] = sites[k - 1] + idx_ranges[k][multi_idx[k]]
    end
    return sites
end

# Compute a correlator for specific absolute site positions per operator piece.
# ops_pv : tuple of original (possibly multi-piece) normalized operators
# op_sites: tuple of Vector{Int}, one per operator, containing absolute sites of each piece.
function _correlator_absolute(state::AbstractMPS, ops_pv::Tuple, op_sites::Tuple,
        backend = DefaultBackend(), allocator = BufferAllocator())
    N = length(ops_pv)

    # Fast path: single operator, single piece.
    if N == 1 && length(op_sites[1]) == 1
        s = op_sites[1][1]
        return _expval_mpotensor(state, _op_pieces(ops_pv[1], s)[1], s, backend, allocator)
    end

    # Build Vₗ starting from first piece of op 1.
    first_site = op_sites[1][1]
    pieces1 = _op_pieces(ops_pv[1], first_site)
    S₁ = _firstspace(pieces1[1])
    @plansor backend=backend allocator=allocator Vₗ[-1 -2; -3] := state.AC[first_site][2 3; -3] *
        removeunit(pieces1[1], 1)[1; 3 -2] * conj(state.AC[first_site][2 1; -1])
    ctr = first_site + 1

    if N == 1
        # Single multi-piece operator: all pieces except last go through Vₗ, then close.
        Vₗ_mid = Vₗ
        for j in 2:(length(op_sites[1]) - 1)
            site = op_sites[1][j]
            Vₗ_mid = site > ctr ? Vₗ_mid * TransferMatrix(state.AR[ctr:(site - 1)], backend, allocator) : Vₗ_mid
            Vₗ_mid = _transfer_right_mpo(Vₗ_mid, pieces1[j], state.AR[site], backend, allocator)
            ctr = site + 1
        end
        last_site = op_sites[1][end]
        Vₗ_mid = last_site > ctr ? Vₗ_mid * TransferMatrix(state.AR[ctr:(last_site - 1)], backend, allocator) : Vₗ_mid
        ON_last = pieces1[end]
        S_last = _lastspace(ON_last)
        S_last == S₁' ||
            throw(ArgumentError("Last operator must end with a trivial right virtual leg."))
        return @plansor backend=backend allocator=allocator Vₗ_mid[1 2; 4] * state.AR[last_site][4 5; 6] *
            removeunit(ON_last, 4)[2 3; 5] * conj(state.AR[last_site][1 3; 6])
    end

    # Remaining pieces of op 1 (for N ≥ 2).
    for j in 2:length(op_sites[1])
        site = op_sites[1][j]
        Vₗ = site > ctr ? Vₗ * TransferMatrix(state.AR[ctr:(site - 1)], backend, allocator) : Vₗ
        Vₗ = _transfer_right_mpo(Vₗ, pieces1[j], state.AR[site], backend, allocator)
        ctr = site + 1
    end

    # Intermediate ops 2..N-1.
    for k in 2:(N - 1)
        pieces_k = _op_pieces(ops_pv[k], op_sites[k][1])
        for (j, site) in enumerate(op_sites[k])
            Vₗ = site > ctr ? Vₗ * TransferMatrix(state.AR[ctr:(site - 1)], backend, allocator) : Vₗ
            Vₗ = _transfer_right_mpo(Vₗ, pieces_k[j], state.AR[site], backend, allocator)
            ctr = site + 1
        end
    end

    # Last op: apply all pieces except the final one, then close with removeunit.
    piecesN = _op_pieces(ops_pv[N], op_sites[N][1])
    Vₗ_mid = Vₗ
    for j in 1:(length(op_sites[N]) - 1)
        site = op_sites[N][j]
        Vₗ_mid = site > ctr ? Vₗ_mid * TransferMatrix(state.AR[ctr:(site - 1)], backend, allocator) : Vₗ_mid
        Vₗ_mid = _transfer_right_mpo(Vₗ_mid, piecesN[j], state.AR[site], backend, allocator)
        ctr = site + 1
    end
    last_site = op_sites[N][end]
    Vₗ_mid = last_site > ctr ? Vₗ_mid * TransferMatrix(state.AR[ctr:(last_site - 1)], backend, allocator) : Vₗ_mid
    ON_last = piecesN[end]
    S_last = _lastspace(ON_last)
    S_last == S₁' ||
        throw(ArgumentError("Last operator must end with a trivial right virtual leg."))
    return @plansor backend=backend allocator=allocator Vₗ_mid[1 2; 4] * state.AR[last_site][4 5; 6] *
        removeunit(ON_last, 4)[2 3; 5] * conj(state.AR[last_site][1 3; 6])
end


"""
    correlator(ψ, O1, O2, i, j)
    correlator(ψ, O12, i, j)

Compute the 2-point correlator <ψ|O1[i]O2[j]|ψ> for inserting `O1` at `i` and `O2` at `j`.
Also accepts ranges for `j`.
"""
function correlator_old end

function correlator_old(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int, j::Int)
    return first(correlator_old(state, O₁, O₂, i, j:j))
end

function correlator_old(
        state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int, js::AbstractRange{Int}
    )
    first(js) > i || @error "i should be smaller than j ($i, $(first(js)))"
    S₁ = _firstspace(O₁)
    isunitspace(S₁) || throw(ArgumentError("O₁ should start with a trivial leg."))
    S₂ = _lastspace(O₂)
    S₂ == S₁' || throw(ArgumentError("O₂ should end with a trivial leg."))

    G = similar(js, scalartype(state))

    @plansor Vₗ[-1 -2; -3] := state.AC[i][2 3; -3] * removeunit(O₁, 1)[1; 3 -2] *
        conj(state.AC[i][2 1; -1])
    ctr = i + 1

    for (k, j) in enumerate(js)
        if j > ctr
            Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
        end
        G[k] = @plansor Vₗ[1 2; 4] * state.AR[j][4 5; 6] * removeunit(O₂, 4)[2 3; 5] *
            conj(state.AR[j][1 3; 6])
        ctr = j
    end
    return G
end

function correlator_old(
        state::AbstractMPS, O₁₂::AbstractTensorMap{<:Any, S, 2, 2}, i::Int, j
    ) where {S}
    O₁, O₂ = decompose_localmpo(add_util_leg(O₁₂))
    return correlator_old(state, O₁, O₂, i, j)
end
