abstract type OptimizerCache end

abstract type Optimizer{T} end

abstract type TDVP{T} <: Optimizer{T} end

mutable struct TDVPCache{T} <: OptimizerCache
    #Ensemble averages:
    L∂L::Array{T,4}
    ΔLL::Array{T,4}

    #Sums:
    mlL::T
    mlL2::T
    acceptance::Float64#UInt64

    #Gradient:
    ∇::Array{T,4}

    # Metric tensor:
    S::Array{T,2}
    avg_G::Array{T}
end

"""
function Base.copy(cache::TDVPCache)
    copy_cache = TDVPCache(
        deepcopy(L∂L),
        deepcopy(ΔLL),
        copy(mlL),
        copy(mlL2),
        0.0,
        deepcopy(∇),
        deepcopy(S),
        zeros(T, 4*params.χ^2*params.uc_size)
    )  
"""

function TDVPCache(A::Array{T,4},params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=TDVPCache(
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        convert(T, 0),
        convert(T, 0),
        0.0,
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        zeros(T, 4*params.χ^2*params.uc_size, 4*params.χ^2*params.uc_size),
        zeros(T, 4*params.χ^2*params.uc_size)
    )  
    return cache
end

mutable struct TDVPl1{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
    #A::Array{T,3}
    mpo::MPO{T}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::TDVPCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    τ::Float64
    ϵ_shift::Float64
    ϵ_SNR::Float64
    ϵ_tol::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

function Base.display(opt::TDVPl1)
    println("\nOptimizer TDVPl1:")
    println("ising_op\t", opt.ising_op)
    println("dephasing_op\t", opt.dephasing_op)
    println("τ\t\t\t", opt.τ)
    println("ϵ_shift\t\t", opt.ϵ_shift)
    println("ϵ_SNR\t\t", opt.ϵ_SNR)
    println("ϵ_tol\t\t", opt.ϵ_tol)
end

function Base.deepcopy(opt::TDVPl1)
    sampler = MetropolisSampler(opt.sampler.N_MC, opt.sampler.burn, opt.sampler.sweeps, opt.params) 
    return TDVPl1(deepcopy(opt.mpo), sampler, opt.optimizer_cache, opt.l1, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
    #return TDVPl1(deepcopy(opt.mpo), opt.sampler, TDVPCache(deepcopy(opt.mpo.A),opt.params), opt.l1, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, set_workspace(deepcopy(opt.mpo.A), opt.params))
    #return TDVPl1(opt.mpo, opt.sampler, opt.optimizer_cache, opt.l1, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
end

function Base.deepcopy(opt::TDVPl1, N_MC_H::Int64)
    Heun_sampler = MetropolisSampler(N_MC_H, opt.sampler.burn, opt.sampler.sweeps, opt.params) 
    return TDVPl1(deepcopy(opt.mpo), Heun_sampler, opt.optimizer_cache, opt.l1, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
    #return TDVPl1(deepcopy(opt.mpo), Heun_sampler, TDVPCache(deepcopy(opt.mpo.A),opt.params), opt.l1, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, set_workspace(deepcopy(opt.mpo.A), opt.params))
    #return TDVPl1(opt.mpo, Heun_sampler, opt.optimizer_cache, opt.l1, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
end

export TDVP

function TDVP(sampler::MetropolisSampler, mpo::MPO{T}, l1::Matrix{T}, τ::Float64, ϵ_shift::Float64, ϵ_SNR::Float64, ϵ_tol::Float64, params::Parameters, ising_int::String) where {T<:Complex{<:AbstractFloat}} 
    if ising_int=="Ising" 
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, Ising(), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    elseif ising_int=="LRIsing"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, LongRangeIsing(params), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    elseif ising_int=="CompetingIsing"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, CompetingIsing(params), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    elseif ising_int=="SquareIsing"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, SquareIsing(), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    elseif ising_int=="CompetingSquareIsing"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, CompetingSquareIsing(params), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    elseif ising_int=="TriangularIsing"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, TriangularIsing(), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end

mutable struct TDVPl2{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
    #A::Array{T,3}
    mpo::MPO{T}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::TDVPCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}
    l2::Array{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    τ::Float64
    ϵ_shift::Float64
    ϵ_SNR::Float64
    ϵ_tol::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

function Base.display(opt::TDVPl2)
    println("\nOptimizer TDVPl2:")
    println("ising_op\t", opt.ising_op)
    println("dephasing_op\t", opt.dephasing_op)
    println("τ\t\t\t", opt.τ)
    println("ϵ_shift\t\t", opt.ϵ_shift)
    println("ϵ_SNR\t\t", opt.ϵ_SNR)
    println("ϵ_tol\t\t", opt.ϵ_tol)
end

export TDVP

function TDVP(sampler::MetropolisSampler, mpo::MPO{T}, l1::Matrix{T}, l2::Array{T}, τ::Float64, ϵ_shift::Float64, ϵ_SNR::Float64, ϵ_tol::Float64, params::Parameters, ising_int::String) where {T<:Complex{<:AbstractFloat}} 
    if ising_int=="Ising" 
        optimizer = TDVPl2(mpo, sampler, TDVPCache(mpo.A, params), l1, l2, Ising(), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end

function Base.deepcopy(opt::TDVPl2)
    return TDVPl2(deepcopy(opt.mpo), opt.sampler, opt.optimizer_cache, opt.l1, opt.l2, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
end

function Base.deepcopy(opt::TDVPl2, N_MC_H::Int64)
    Heun_sampler = MetropolisSampler(N_MC_H, 0, opt.sweeps, opt.params) 
    return TDVPl2(deepcopy(opt.mpo), Heun_sampler, opt.optimizer_cache, opt.l1, opt.l2, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
end

mutable struct TDVP_H{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
    #A::Array{T,3}
    mpo::MPO{T}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::TDVPCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}
    l2::Array{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    τ::Float64
    ϵ_shift::Float64
    ϵ_SNR::Float64
    ϵ_tol::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

function Base.display(opt::TDVP_H)
    println("\nOptimizer TDVP_H:")
    println("ising_op\t", opt.ising_op)
    println("dephasing_op\t", opt.dephasing_op)
    println("τ\t\t\t", opt.τ)
    println("ϵ_shift\t\t", opt.ϵ_shift)
    println("ϵ_SNR\t\t", opt.ϵ_SNR)
    println("ϵ_tol\t\t", opt.ϵ_tol)
end

export TDVP_H

function TDVP_H(sampler::MetropolisSampler, mpo::MPO{T}, l1::Matrix{T}, l2::Array{T}, τ::Float64, ϵ_shift::Float64, ϵ_SNR::Float64, ϵ_tol::Float64, params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    optimizer = TDVP_H(mpo, sampler, TDVPCache(mpo.A, params), l1, l2, SquareIsing(), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    return optimizer
end

function Base.deepcopy(opt::TDVP_H)
    return TDVP_H(deepcopy(opt.mpo), opt.sampler, opt.optimizer_cache, opt.l1, opt.l2, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
end

function Base.deepcopy(opt::TDVP_H, N_MC_H::Int64)
    Heun_sampler = MetropolisSampler(N_MC_H, 0, opt.sweeps, opt.params) 
    return TDVP_H(deepcopy(opt.mpo), Heun_sampler, opt.optimizer_cache, opt.l1, opt.l2, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
end



### EDIT LATER:

"""

abstract type ExactTDVP{T} <: Optimizer{T} end

mutable struct TI_ExactTDVPCache{T} <: OptimizerCache
    #Ensemble averages:
    L∂L::Array{T,4}
    ΔLL::Array{T,4}

    Z::Float64

    #Sums:
    mlL::T

    #Gradient:
    ∇::Array{T,4}

    # Metric tensor:
    S::Array{T,2}
    avg_G::Array{T}
end

function TI_ExactTDVPCache(A::Array{T,4},params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=TI_ExactTDVPCache(
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        0.0,
        convert(T,0),
        zeros(T,params.uc_size,params.χ,params.χ,4),
        zeros(T,4*params.χ^2,4*params.χ^2),
        zeros(T,4*params.χ^2)
    )  
    return cache
end

mutable struct TI_ExactTDVPl1{T<:Complex{<:AbstractFloat}} <: ExactTDVP{T}

    #MPO:
    mpo::MPO{T}

    #Basis:
    basis::Basis

    #Optimizer:
    optimizer_cache::TI_ExactTDVPCache{T}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}

end

function TDVP(basis::Basis, mpo::MPO{T}, l1::Matrix{T}, ϵ::Float64, params::Parameters, ising_int::String) where {T<:Complex{<:AbstractFloat}} 
    if ising_int=="Ising" 
        optimizer = TI_ExactTDVPl1(mpo, basis, TI_ExactTDVPCache(mpo.A, params), l1, Ising(), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    elseif ising_int=="LRIsing"
        optimizer = TI_ExactTDVPl1(mpo, basis, TI_ExactTDVPCache(mpo.A, params), l1, LongRangeIsing(params), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end
"""