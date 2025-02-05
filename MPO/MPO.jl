export NormalizeMPO!

export MPO

abstract type MatrixProductOperator{T<:Complex{<:AbstractFloat}} end

mutable struct MPO{T} <: MatrixProductOperator{T}
    A::Array{T,4}
end

function Base.similar(mpo::MPO)
    return MPO(similar(mpo.A))
end

function Base.deepcopy(mpo::MPO)
    return MPO(deepcopy(mpo.A))
end

function trMPO(params::Parameters, sample::Projector, mpo::MPO{T}) where {T<:Complex{<:AbstractFloat}}

    A = mpo.A
    trMPO=Matrix{ComplexF64}(I, params.χ, params.χ)
    for i = 1:params.N
        n = mod1(i, params.uc_size)
        trMPO*=A[n,:,:,idx(sample,i)]
    end
    return tr(trMPO)::ComplexF64
end

function L_MPO_products!(L_set, sample::Projector, mpo::MPO{T}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}}
    
    A = mpo.A
    L_set[1] = cache.ID
    for i = 1:params.N
        n = mod1(i, params.uc_size)
        mul!(L_set[i+1], L_set[i], @view(A[n,:,:,idx(sample,i)]))
    end
    return L_set
end

function R_MPO_products!(R_set, sample::Projector, mpo::MPO{T}, params::Parameters, cache::Workspace) where {T<:Complex{<:AbstractFloat}}

    A = mpo.A
    R_set[1] = cache.ID
    for i = params.N:-1:1
        n = mod1(i, params.uc_size)
        mul!(R_set[params.N+2-i], @view(A[n,:,:,idx(sample,i)]), R_set[params.N+1-i])
    end
    return R_set
end

function ∂MPO(sample::Projector, L_set::Vector{<:Matrix{T}}, R_set::Vector{<:Matrix{T}}, params::Parameters, cache::Workspace, mpo::MPO{T}) where {T<:Complex{<:AbstractFloat}}

    cache.∂ = zeros(T, params.uc_size, params.χ, params.χ, 4)
    for m = 1:params.N
        mul!(cache.B, R_set[params.N+1-m], L_set[m])
        n = mod1(m, params.uc_size)
        for i=1:params.χ
            for j=1:params.χ
                cache.∂[n, i, j, idx(sample,m)] += cache.B[j, i]
            end
        end
    end
    return cache.∂
end