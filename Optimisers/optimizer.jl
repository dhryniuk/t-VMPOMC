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
    acceptance::Float64

    #Gradient:
    ∇::Array{T,4}

    # Metric tensor:
    S::Array{T,2}
end

function TDVPCache(A::Array{T,4},params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=TDVPCache(
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        convert(T, 0),
        convert(T, 0),
        0.0,
        zeros(T, params.uc_size, params.χ, params.χ, 4),
        zeros(T, 4*params.χ^2*params.uc_size, 4*params.χ^2*params.uc_size),
    )  
    return cache
end

mutable struct TDVPl1{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
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
    workspace::Workspace{T}
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
end

function Base.deepcopy(opt::TDVPl1, N_MC_H::Int64)
    Heun_sampler = MetropolisSampler(N_MC_H, opt.sampler.burn, opt.sampler.sweeps, opt.params) 
    return TDVPl1(deepcopy(opt.mpo), Heun_sampler, opt.optimizer_cache, opt.l1, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
end

export TDVP

function TDVP(sampler::MetropolisSampler, mpo::MPO{T}, l1::Matrix{T}, τ::Float64, ϵ_shift::Float64, ϵ_SNR::Float64, ϵ_tol::Float64, params::Parameters, ising_int::String) where {T<:Complex{<:AbstractFloat}} 
    if ising_int=="Ising" 
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, Ising(), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    elseif ising_int=="LRIsing"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, LongRangeIsing(params), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    elseif ising_int=="LongRangeRydberg"
        optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), l1, LongRangeRydberg(params), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
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
    workspace::Workspace{T}
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

mutable struct TDVPXYZ{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
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
    workspace::Workspace{T}
end

function Base.display(opt::TDVPXYZ)
    println("\nOptimizer TDVPXYZ:")
    println("ising_op\t", opt.ising_op)
    println("dephasing_op\t", opt.dephasing_op)
    println("τ\t\t\t", opt.τ)
    println("ϵ_shift\t\t", opt.ϵ_shift)
    println("ϵ_SNR\t\t", opt.ϵ_SNR)
    println("ϵ_tol\t\t", opt.ϵ_tol)
end

export TDVPXYZ

function TDVPXYZ(sampler::MetropolisSampler, mpo::MPO{T}, l1::Matrix{T}, l2::Array{T}, τ::Float64, ϵ_shift::Float64, ϵ_SNR::Float64, ϵ_tol::Float64, params::Parameters, ising_int::String) where {T<:Complex{<:AbstractFloat}} 
    if ising_int=="CompetingIsing" 
        optimizer = TDVPXYZ(mpo, sampler, TDVPCache(mpo.A, params), l1, l2, CompetingIsing(params), LocalDephasing(), params, τ, ϵ_shift, ϵ_SNR, ϵ_tol, set_workspace(mpo.A, params))
    else
        error("Unrecognized Ising interaction")
    end
    return optimizer
end

function Base.deepcopy(opt::TDVPXYZ)
    return TDVPXYZ(deepcopy(opt.mpo), opt.sampler, opt.optimizer_cache, opt.l1, opt.l2, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
end

function Base.deepcopy(opt::TDVPXYZ, N_MC_H::Int64)
    Heun_sampler = MetropolisSampler(N_MC_H, 0, opt.sweeps, opt.params) 
    return TDVPXYZ(deepcopy(opt.mpo), Heun_sampler, opt.optimizer_cache, opt.l1, opt.l2, opt.ising_op, opt.dephasing_op, opt.params, opt.τ, opt.ϵ_shift, opt.ϵ_SNR, opt.ϵ_tol, opt.workspace)
end

mutable struct TDVP_H{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
    mpo::MPO{T}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::TDVPCache{T}

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
    workspace::Workspace{T}
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