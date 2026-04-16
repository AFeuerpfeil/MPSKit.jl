"""
$(TYPEDEF)

Single site infinite DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct IDMRG{A} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol

    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter

    miniter::Int = 0

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity

    "algorithm used for gauging the MPS"
    alg_gauge = Defaults.alg_gauge()


    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A = Defaults.alg_eigsolve()

    expscheme::TruncationStrategy = noexpand()

    trscheme::TruncationStrategy = notrunc()
end

"""
$(TYPEDEF)

Two-site infinite DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct IDMRG2{A, S} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol

    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter

    miniter::Int = 0

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity

    "algorithm used for gauging the MPS"
    alg_gauge = Defaults.alg_gauge()

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A = Defaults.alg_eigsolve()

    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    expscheme::TruncationStrategy = noexpand()

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy) of the two-site update"
    trscheme::TruncationStrategy = notrunc()
end


# Internal state of the IDMRG algorithm
struct IDMRGState{S, O, E, T}
    mps::S
    operator::O
    envs::E
    iter::Int
    ϵ::Float64 # TODO: Could be any <:Real
    energy::T
end
function IDMRGState{T}(mps::S, operator::O, envs::E, iter::Int, ϵ::Float64, energy) where {S, O, E, T}
    return IDMRGState{S, O, E, T}(mps, operator, envs, iter, ϵ, T(energy))
end

function find_groundstate!(
        ::InfiniteChainStyle, mps::S, operator, alg::alg_type,
        envs = environments(mps, operator)
    ) where {alg_type <: Union{<:IDMRG, <:IDMRG2}, S}
    (length(mps) ≤ 1 && alg isa IDMRG2) && throw(ArgumentError("unit cell should be >= 2"))
    log = alg isa IDMRG ? IterLog("IDMRG") : IterLog("IDMRG2")
    iter = 0
    ϵ = calc_galerkin(mps, operator, mps, envs)
    E = zero(TensorOperations.promote_contract(scalartype(mps), scalartype(operator)))

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 begin
            E = expectation_value(mps, operator, envs)
            loginit!(log, ϵ, E)
        end
    end

    state = IDMRGState(mps, operator, envs, iter, ϵ, E)
    it = IterativeSolver(alg, state)

    return LoggingExtras.withlevel(; alg.verbosity) do
        for (mps, envs, ϵ, ΔE) in it
            if ϵ ≤ alg.tol && it.iter > alg.miniter
                @infov 2 logfinish!(log, it.iter, ϵ, ΔE)
                break
            end
            if it.iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, it.iter, ϵ, ΔE)
                break
            end
            @infov 3 logiter!(log, it.iter, ϵ, ΔE)
        end

        alg_gauge = updatetol(alg.alg_gauge, it.iter, it.ϵ)
        ψ′ = S.name.wrapper(it.state.mps.AR; alg_gauge.tol, alg_gauge.maxiter)

        envs = recalculate!(it.state.envs, ψ′, it.state.operator, ψ′)
        return ψ′, envs, it.state.ϵ
    end
end

function Base.iterate(
        it::IterativeSolver{alg_type}, state::IDMRGState{<:Any, <:Any, <:Any, T} = it.state
    ) where {alg_type <: Union{<:IDMRG, <:IDMRG2}, T}
    mps, envs, C_old, E_new = localupdate_step!(it, state)

    # error criterion
    ϵ = bond_error(C_old, mps.C[0])

    # New energy
    ΔE = (E_new - state.energy) / 2
    (alg_type <: IDMRG2 && length(mps) == 2) && (ΔE /= 2) # This extra factor gives the correct energy per unit cell. I have no clue why right now.

    # update state
    it.state = IDMRGState{T}(mps, state.operator, envs, state.iter + 1, ϵ, E_new)

    return (mps, envs, ϵ, ΔE), it.state
end

function localupdate_step!(
        it::IterativeSolver{<:IDMRG}, state
    )
    alg_eigsolve = updatetol(it.alg_eigsolve, state.iter, state.ϵ)
    expscheme = updatetruncation(it.expscheme; iter = state.iter, current_rank = maxlinkdim(state.mps))
    trscheme = updatetruncation(it.trscheme; iter = state.iter)
    return _localupdate_sweep_idmrg!(state.mps, state.operator, state.envs, alg_eigsolve, trscheme, expscheme)
end

function localupdate_step!(
        it::IterativeSolver{<:IDMRG2}, state
    )
    alg_eigsolve = updatetol(it.alg_eigsolve, state.iter, state.ϵ)
    expscheme = updatetruncation(it.expscheme; iter = state.iter, current_rank = maxlinkdim(state.mps))
    trscheme = updatetruncation(it.trscheme; iter = state.iter)
    return _localupdate_sweep_idmrg2!(state.mps, state.operator, state.envs, alg_eigsolve, trscheme, it.alg_svd, expscheme)
end

function _localupdate_sweep_idmrg!(ψ, H, envs, alg_eigsolve, alg_trscheme, expscheme)
    local E
    C_old = ψ.C[0]
    # left to right sweep
    for pos in 1:length(ψ)
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        ψ.AL[pos], ψ.C[pos] = left_orth!(ψ.AC[pos]; trunc = alg_trscheme)
        ψ.AL[pos], (ψ.C[pos], ψ.AC[pos + 1], ψ.AL[pos + 1]) = changebonds_left(ψ.AL[pos], (ψ.C[pos], ψ.AC[pos + 1], ψ.AL[pos + 1]), expscheme)
        if pos == length(ψ) # AC needed in next sweep
            ψ.AC[pos] = _mul_tail(ψ.AL[pos], ψ.C[pos])
        end
        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
    end

    # right to left sweep
    for pos in length(ψ):-1:1
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        E, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)

        C, temp = right_orth!(_transpose_tail(ψ.AC[pos]); trunc = alg_trscheme)
        (ψ.C[pos - 1], ψ.AR[pos - 1], ψ.AC[pos - 1]), temp = changebonds_right((C, ψ.AR[pos - 1],ψ.AC[pos - 1]), temp, expscheme)
        ψ.AR[pos] = _transpose_front(temp)
        if pos == 1 # AC needed in next sweep
            ψ.AC[pos] = _mul_front(ψ.C[pos - 1], ψ.AR[pos])
        end

        transfer_rightenv!(envs, ψ, H, ψ, pos - 1)
    end
    return ψ, envs, C_old, E
end


function _localupdate_sweep_idmrg2!(ψ, H, envs, alg_eigsolve, alg_trscheme, alg_svd, expscheme)
    # sweep from left to right
    for pos in 1:(length(ψ) - 1)
        ac2 = AC2(ψ, pos; kind = :ACAR)
        h_ac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
        _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

        al, c, ar = svd_trunc!(ac2′; trunc = alg_trscheme, alg = alg_svd)
        al, (c, ψ.AL[pos + 1]) = changebonds_left(al, (c, ψ.AL[pos + 1]), expscheme; ac2=ac2)
        normalize!(c)

        ψ.AL[pos] = al
        ψ.C[pos] = complex(c)
        ψ.AR[pos + 1] = _transpose_front(ar)
        ψ.AC[pos + 1] = _transpose_front(c * ar)

        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
        transfer_rightenv!(envs, ψ, H, ψ, pos)
    end

    # update the edge
    ψ.AL[end] = ψ.AC[end] / ψ.C[end]
    ψ.AC[1] = _mul_tail(ψ.AL[1], ψ.C[1])
    ac2 = AC2(ψ, length(ψ); kind = :ALAC)
    h_ac2 = AC2_hamiltonian(length(ψ), ψ, H, ψ, envs)
    _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

    al, c, ar = svd_trunc!(ac2′; trunc = alg_trscheme, alg = alg_svd)
    al, c, ar = changebonds(al, c, ar, expscheme; ac2=ac2)
    normalize!(c)

    ψ.AL[end] = al
    ψ.C[end] = complex(c)
    ψ.AR[end+1] = _transpose_front(ar)

    ψ.AC[end] = _mul_tail(al, c)
    ψ.AC[end+1] = _transpose_front(c * ar)
    ψ.AL[end+1] = ψ.AC[end+1] / ψ.C[end+1]

    C_old = complex(c)

    # update environments
    transfer_leftenv!(envs, ψ, H, ψ, 1)
    transfer_rightenv!(envs, ψ, H, ψ, 0)

    # sweep from right to left
    for pos in (length(ψ) - 1):-1:1
        ac2 = AC2(ψ, pos; kind = :ALAC)
        h_ac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
        _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

        al, c, ar = svd_trunc!(ac2′; trunc = alg_trscheme, alg = alg_svd)
        (c, ψ.AR[pos]), ar = changebonds_right((c, ψ.AR[pos]), ar, expscheme; ac2=ac2)
        normalize!(c)

        ψ.AL[pos] = al
        ψ.AC[pos] = _mul_tail(al, c)
        ψ.C[pos] = complex(c)
        ψ.AR[pos + 1] = _transpose_front(ar)
        ψ.AC[pos + 1] = _transpose_front(c * ar)

        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
        transfer_rightenv!(envs, ψ, H, ψ, pos)
    end

    # update the edge
    ψ.AC[0] = _mul_front(ψ.C[- 1], ψ.AR[0])
    ψ.AR[1] = _transpose_front(ψ.C[0] \ _transpose_tail(ψ.AC[1]))
    ac2 = AC2(ψ, 0; kind = :ACAR)
    h_ac2 = AC2_hamiltonian(0, ψ, H, ψ, envs)
    E, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)
    al, c, ar = svd_trunc!(ac2′; trunc = alg_trscheme, alg = alg_svd)
    al, c, ar = changebonds(al, c, ar, expscheme; ac2=ac2)
    normalize!(c)

    ψ.AL[0] = al
    ψ.C[0] = complex(c)
    ψ.AR[1] = _transpose_front(ar)

    ψ.AR[0] = _transpose_front(ψ.C[-1] \ _transpose_tail(al * c))
    ψ.AC[1] = _transpose_front(c * ar)

    transfer_leftenv!(envs, ψ, H, ψ, 1)
    transfer_rightenv!(envs, ψ, H, ψ, 0)
    return ψ, envs, C_old, E
end
