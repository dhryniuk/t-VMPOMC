# This file contains routines required for the computation of purely-diagonal interaction terms of the Lindbladian

abstract type DiagonalOperators end

abstract type IsingInteraction <: DiagonalOperators end

struct Ising <: IsingInteraction end

struct SquareIsing <: IsingInteraction end

struct TriangularIsing <: IsingInteraction end

struct LongRangeIsing <: IsingInteraction
    α::Float64
    Kac_norm::Float64
end

struct LongRangeRydberg <: IsingInteraction
    α::Float64
    Kac_norm::Float64
end

struct CompetingIsing <: IsingInteraction
    α1::Float64
    Kac_norm1::Float64
    α2::Float64
    Kac_norm2::Float64
end

struct CompetingSquareIsing <: IsingInteraction
    α1::Float64
    Kac_norm1::Float64
    α2::Float64
    Kac_norm2::Float64
end

function HarmonicNumber(n::Int,α::Float64)
    h=0
    for i in 1:n
        h+=i^(-α)
    end
    return h
end

function Kac_norm(N, α)
    if mod(N,2)==0
        return (2*HarmonicNumber(1+N÷2,α) - 1 - (1+N÷2)^(-α))
    else
        return (2*HarmonicNumber(1+(N-1)÷2,α) - 1)
    end
end

function LongRangeIsing(params::Parameters)
    α = params.α1
    #K = 1
    K = Kac_norm(params.N, params.α1)
    return LongRangeIsing(α,K)
end

function LongRangeRydberg(params::Parameters)
    α = params.α1
    #K = 1
    K = Kac_norm(params.N, params.α1)
    return LongRangeRydberg(α,K)
end

function CompetingIsing(params::Parameters)
    α1 = params.α1
    K1 = Kac_norm(params.N, params.α1)
    α2 = params.α2
    K2 = Kac_norm(params.N, params.α2)
    return CompetingIsing(α1,K1,α2,K2)
end

function CompetingSquareIsing(params::Parameters)
    α1 = params.α1
    K1 = Kac_norm(params.N, params.α1)
    α2 = params.α2
    K2 = Kac_norm(params.N, params.α2)
    return CompetingSquareIsing(α1,K1,α2,K2)
end

abstract type Dephasing <: DiagonalOperators end

#struct NoDephasing <: Dephasing end

struct LocalDephasing <: Dephasing end

struct CollectiveDephasing <: Dephasing end
