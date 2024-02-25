abstract type OptimizerCache end

abstract type Optimizer{T} end

abstract type TDVP{T} <: Optimizer{T} end

mutable struct TDVPCache{T} <: OptimizerCache
    #Ensemble averages:
    L∂L::Array{T,4}
    ΔLL::Array{T,4}

    #Sums:
    mlL::T
    acceptance::Float64#UInt64

    #Gradient:
    ∇::Array{T,4}

    # Metric tensor:
    S::Array{T,2}
    avg_G::Array{T}
end

function TDVPCache(A::Array{T,4},params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=TDVPCache(
        zeros(T, params.N, params.χ, params.χ, 4),
        zeros(T, params.N, params.χ, params.χ, 4),
        convert(T,0),
        0.0,#convert(UInt64,0),
        zeros(T,params.N,params.χ,params.χ,4),
        zeros(T,4*params.χ^2*params.N,4*params.χ^2*params.N),
        zeros(T,4*params.χ^2*params.N)
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
    list_l1::Vector{Matrix{T}}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

function TDVP(sampler::MetropolisSampler, mpo::MPO{T}, list_l1::Vector{Matrix{T}}, ϵ::Float64, params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    optimizer = TDVPl1(mpo, sampler, TDVPCache(mpo.A, params), list_l1, Ising(), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    return optimizer
end


mutable struct TI_TDVPCache{T} <: OptimizerCache
    #Ensemble averages:
    L∂L::Array{T,3}
    ΔLL::Array{T,3}

    #Sums:
    mlL::T
    acceptance::Float64#UInt64

    #Gradient:
    ∇::Array{T,3}

    # Metric tensor:
    S::Array{T,2}
    avg_G::Array{T}
end

function TI_TDVPCache(A::Array{T,3},params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=TI_TDVPCache(
        zeros(T,params.χ,params.χ,4),
        zeros(T,params.χ,params.χ,4),
        convert(T,0),
        0.0,#convert(UInt64,0),
        zeros(T,params.χ,params.χ,4),
        zeros(T,4*params.χ^2,4*params.χ^2),
        zeros(T,4*params.χ^2)
    )  
    return cache
end

mutable struct TI_TDVPl1{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
    #A::Array{T,3}
    mpo::TI_MPO{T}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::TI_TDVPCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Diagonal operators:
    ising_op::IsingInteraction
    dephasing_op::Dephasing

    #Parameters:
    params::Parameters
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

function TDVP(sampler::MetropolisSampler, mpo::TI_MPO{T}, l1::Matrix{T}, ϵ::Float64, params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    optimizer = TI_TDVPl1(mpo, sampler, TI_TDVPCache(mpo.A, params), l1, Ising(), LocalDephasing(), params, ϵ, set_workspace(mpo.A, params))
    return optimizer
end