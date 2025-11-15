"""
    struct FiniteEnvironments <: AbstractMPSEnvironments

Environment manager for `FiniteMPS` and `WindowMPS`. This structure is responsible for automatically checking
if the queried environment is still correctly cached and if not recalculates.
"""
struct FiniteEnvironments{A, B, C, D} <: AbstractMPSEnvironments
    above::A

    operator::B #the operator

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}

    GLs::Vector{D}
    GRs::Vector{D}
end

GeometryStyle(::Type{FiniteEnvironments}) = FiniteStyle()

function environments(below, operator, leftstart, rightstart)
    return environments(below, operator, nothing, leftstart, rightstart)
end
function environments(below, operator, above, leftstart, rightstart)
    leftenvs = [i == 0 ? leftstart : similar(leftstart) for i in 0:length(below)]
    N = length(below)
    rightenvs = [i == N ? rightstart : similar(rightstart) for i in 0:length(below)]

    t = similar(below.AL[1])
    return FiniteEnvironments(
        above, operator, fill(t, length(below)),
        fill(t, length(below)),
        leftenvs,
        rightenvs
    )
end

function environments(
        ::FiniteStyle, ::OperatorStyle,
        below::AbstractMPS, O::AbstractMPO, above::Union{AbstractMPS, Nothing}
    )
    Vl_bot = left_virtualspace(below, 1)
    Vl_mid = left_virtualspace(O, 1)
    Vl_top = left_virtualspace(above, 1)
    leftstart = isomorphism(storagetype(below), Vl_bot ⊗ Vl_mid' ← Vl_top)

    N = length(below)
    Vr_bot = right_virtualspace(below, N)
    Vr_mid = right_virtualspace(O, N)
    Vr_top = right_virtualspace(above, N)
    rightstart = isomorphism(storagetype(below), Vr_top ⊗ Vr_mid ← Vr_bot)

    return environments(below, O, above, leftstart, rightstart)
end

function environments(
        ::WindowStyle, ::OperatorStyle,
        below::AbstractMPS, O::AbstractMPO, above::Union{AbstractMPS, Nothing};
        lenvs = environments(below.left_gs, O),
        renvs = environments(below.right_gs, O)
    )
    leftstart = copy(lenvs.GLs[1])
    rightstart = copy(renvs.GRs[end])

    return environments(below, O, above, leftstart, rightstart)
end

function environments(below::AbstractMPS, above::AbstractMPS)
    return environments(GeometryStyle(below) & GeometryStyle(above), below, above)
end
function environments(::FiniteStyle, below::FiniteMPS, above::FiniteMPS)
    operator = fill(nothing, length(below))
    return environments(below, operator, above, l_LL(above), r_RR(above))
end
function environments(::WindowStyle, below::WindowMPS, above::WindowMPS)
    below.left_gs == above.left_gs ||
        throw(ArgumentError("left gs differs"))
    below.right_gs == above.right_gs ||
        throw(ArgumentError("right gs differs"))

    operator = fill(nothing, length(below))
    return environments(below, operator, above, l_LL(above), r_RR(above))
end

function environments(state::AbstractMPS, operator::ProjectionOperator)
    return environments(GeometryStyle(state), state, operator)
end
function environments(::Union{FiniteStyle, WindowStyle}, state::FiniteMPS, operator::ProjectionOperator)
    @plansor leftstart[-1; -2 -3 -4] := l_LL(operator.ket)[-3; -4] *
        l_LL(operator.ket)[-1; -2]
    @plansor rightstart[-1; -2 -3 -4] := r_RR(operator.ket)[-1; -2] *
        r_RR(operator.ket)[-3; -4]
    return environments(
        state, fill(nothing, length(state)), operator.ket, leftstart,
        rightstart
    )
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
poison!(ca::AbstractMPSEnvironments, ind) = poison!(GeometryStyle(ca), ca, ind)
function poison!(::FiniteStyle, ca::AbstractMPSEnvironments, ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    return ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

function rightenv(::FiniteStyle, ca::AbstractMPSEnvironments, ind, state::AbstractMPS)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind + 1))
    a = isnothing(a) ? nothing : length(state) - a + 1

    if !isnothing(a)
        #we need to recalculate
        for j in a:-1:(ind + 1)
            above = isnothing(ca.above) ? state.AR[j] : ca.above.AR[j]
            ca.GRs[j] = TransferMatrix(above, ca.operator[j], state.AR[j]) *
                ca.GRs[j + 1]
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.GRs[ind + 1]
end

function leftenv(::FiniteStyle, ca::AbstractMPSEnvironments, ind, state::AbstractMPS)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind - 1))

    if !isnothing(a)
        #we need to recalculate
        for j in a:(ind - 1)
            above = isnothing(ca.above) ? state.AL[j] : ca.above.AL[j]
            ca.GLs[j + 1] = ca.GLs[j] *
                TransferMatrix(above, ca.operator[j], state.AL[j])
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.GLs[ind]
end
