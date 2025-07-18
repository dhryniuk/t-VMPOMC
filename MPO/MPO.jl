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

"""
function center_∂MPO!(cache::Workspace, params::Parameters, mpo::MPO)
    # cache.∂ is assumed to be a 4D array with dimensions
    # (uc_size, χ, χ, d) where d is the derivative index
    derv = cache.∂
    uc_size, χ, _, dsize = size(derv)
    # For each block corresponding to the n-th operator and a particular derivative channel,
    # subtract the trace part so that the block becomes traceless.
    for n in 1:uc_size
        # It is efficient to create the identity outside the inner loop.
        Iχ = Matrix{eltype(derv)}(I, χ, χ)
        for k in 1:dsize
            # Get a view of the (χ x χ) derivative matrix.
            D = @view derv[n, :, :, k]
            # Compute the trace
            tr_D = tr(D)
            # Subtract the trace contribution:
            # Note: (tr_D/χ)*Iχ is the component of D proportional to the identity.
            D .-= (tr_D/χ)*Iχ
        end
    end
    # Normalize the MPO (using the NormalizeMPO! function defined elsewhere)
    NormalizeMPO!(mpo)
    return derv
end
"""

export construct_density_matrix

function construct_density_matrix(mpo::MPO{T}, params::Parameters, basis::Basis) where {T<:Complex{<:AbstractFloat}}
    # Construct the density matrix from the MPO
    ρ = zeros(T, 2^params.N, 2^params.N)
    for (i,ket) in enumerate(basis)
        for (j,bra) in enumerate(basis)
            ρ[i,j] = trMPO(params, Projector(ket, bra), mpo)
        end
    end
    return ρ
end