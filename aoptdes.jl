export Aopt
struct Aopt{T <: Real} 
    index::T
end
Aopt() = Aopt(0.0)

using LinearAlgebra, ForwardDiff, MathProgBase, Ipopt, NLopt

struct optModel{T <: LinearAlgebra.BlasReal} <: MathProgBase.AbstractNLPEvaluator
    "`n`: number of design points"
    n::Int
    "`p`: number of model parameters"
    p::Int
    "`q`: dimension of design points"
    q::Int
    "`θ`: model parameters"
    θ::Vector{T}
    "`x`: design points"
    x::Matrix{T}
    "`w`: weights of design points"
    w::Vector{T}
    "`M`: Fisher information matrix"
    M::Matrix{T}
end

function optModel(
    n::Int,
    p::Int,
    q::Int,
    θ::Vector{T}
    ) where T <: LinearAlgebra.BlasFloat
    x = zeros(T, n, q)
    w = zeros(T, n)
    M = zeros(T, p, p)
    optModel{eltype(θ)}(n, p, q, θ, x, w, M)
end

"""
Update Fisher information matrix from design points 
    `x` and regression parameters `θ`.
"""
function update_M!(m::optModel)
    T = eltype(m.x)
    fill!(m.M, 0)
    zi = Array{T}(undef,m.p)    ####change
    for i in 1:m.n
        update_z!(zi, @view m.x[i, :])
        ηi = dot(zi, m.θ)
        BLAS.syrk!('U', 'N', m.w[i] * λ(ηi), zi, one(T), m.M)
    end
    m.M
end

update_z!(dz, z, x) = ForwardDiff.jacobian!(dz, update_z!, z, x)

"""
Mapping from regression systematic component `η` to regression weight `λ`.
"""
# function λ(η::Number)
#     eη = exp(η)
#     eη / (1 + eη) / (1 + eη)
# end
#function λ(η::Number)
#     exp(η)
# end
λderiv = x -> ForwardDiff.derivative(λ, x)

function find_opt!(s::Apot{<:Real}, m::optModel,
    solver = NLoptSolver(algorithm=:GN_ISRES), 
    localalgo::Symbol = :LN_COBYLA)

  function aobjective(m::optModel)
    update_M!(m)
    T = eltype(m.x)
    Minv = inv(Symmetric(m.M))
    obj = tr(Minv)   
  end

  function find_aopt!(
    m::optModel,
    solver = NLoptSolver(algorithm=:GN_ISRES), 
    localalgo::Symbol = :LN_COBYLA
    )
    T = eltype(m.x)
    # set up optimization problem
    prob = MathProgBase.NonlinearModel(solver)
    l = [fill(T(-1), m.q * m.n); fill(T(0), m.n)]
    u = fill(T(1), (m.q + 1) * m.n)
    lb = [one(T)]
    ub = [one(T)]
    MathProgBase.loadproblem!(prob, (m.q + 1) * m.n, 1, l, u, lb, ub, :Min, m)
    # set local algorithm for Augmented Lagrangian Method
    if isa(solver, NLopt.NLoptSolver) && solver.algorithm == :AUGLAG
        localopt = NLopt.Opt(localalgo, (m.q + 1) * m.n)
        ftol_rel!(localopt, 1e-8)
        xtol_rel!(localopt, 1e-8)
        NLopt.local_optimizer!(prob.opt, localopt)
    end
    # initialize
    par0 = [m.x[:]; m.w[:]]
    MathProgBase.setwarmstart!(prob, par0)
    # optimize
    MathProgBase.optimize!(prob)
    # ouput
    stat = MathProgBase.status(prob)
    stat == :Optimal ||warn("Optimization unsuccesful; got $stat")
    xsol = MathProgBase.getsolution(prob)
    copyto!(m.x, 1, xsol, 1, m.q * m.n)     #change
    copyto!(m.w, 1, xsol, m.q * m.n + 1, m.n)    #change
    end


function MathProgBase.initialize(m::optModel,
    requested_features::Vector{Symbol})
    for feat in requested_features
      if !(feat in [:Grad :Jac])
        error("Unsupported feature $feat")
      end
    end
end

MathProgBase.features_available(m::optModel) = [:Grad :Jac]

function MathProgBase.eval_f(m::optModel, par::Vector)
    copyto!(m.x, 1, par, 1, m.q * m.n)    #change
    copyto!(m.w, 1, par, m.q * m.n + 1, m.n)   #change
    aobjective(m)
end

function MathProgBase.eval_grad_f(m::optModel, ∇::Vector, par::Vector)
    copyto!(m.x, 1, par, 1, m.q * m.n)   #change
    copyto!(m.w, 1, par, m.q * m.n + 1, m.n)    #change
    update_M!(m)
    # Mchol = cholfact!(Symmetric(m.M))
    Meig = eigen(Symmetric(m.M))   ###change
    zi = zeros(m.p)
    dzi = zeros(m.p, m.q)
    Minvzi = zeros(m.p)
    tmpv = similar(Minvzi)
    for i in 1:m.n
        update_z!(dzi, zi, @view m.x[i, :])
        ηi = dot(zi, m.θ)
        λi = λ(ηi)
        mul!(tmpv, transpose(Meig.vectors), zi)   ##change
        tmpv ./= Meig.values
        mul!(Minvzi, Meig.vectors, tmpv)     ##change
        # A_ldiv_B!(Minvzi, Mchol, zi)
        zitMinvzi = dot(zi, Minvzi)
        # gradient vs xi
        mul!(view(∇, i:m.n:(m.n * (m.q - 1) + i)), transpose(dzi), 
            m.w[i] * zitMinvzi * λderiv(ηi) * m.θ + m.w[i] * 2λi * Minvzi)     ##change
        # derivative vs wi
        ∇[m.n * m.q + i] = λi * zitMinvzi
    end
    ∇
end

function MathProgBase.eval_g(m::optModel, g::Vector, par::Vector)
    g[1] = sum(@view par[(m.q * m.n + 1):end])
end

function MathProgBase.jac_structure(m::optModel)
    fill(1, m.n), collect((m.q * m.n + 1):(m.q * m.n + m.n))
end

function MathProgBase.eval_jac_g(m::optModel, J::Vector, par::Vector)
    fill!(J, 1)
end

MathProgBase.isobjlinear(m::optModel) = false

MathProgBase.isobjquadratic(m::optModel) = false

MathProgBase.isconstrlinear(m::optModel, i::Int) = true
    
 m
end