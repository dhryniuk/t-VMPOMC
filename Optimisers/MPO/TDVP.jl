export TDVP, ComputeGradient!, MPI_mean!, Optimize!


mutable struct TDVPCache{T} <: OptimizerCache
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

function TDVPCache(A::Array{T,3},params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    cache=TDVPCache(
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

mutable struct TDVPl1{T<:Complex{<:AbstractFloat}} <: TDVP{T}

    #MPO:
    A::Array{T,3}

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
    ϵ::Float64

    #Workspace:
    workspace::Workspace{T}#Union{workspace,Nothing}

end

function TDVP(sampler::MetropolisSampler, A::Array{T,3}, l1::Matrix{T}, ϵ::Float64, params::Parameters) where {T<:Complex{<:AbstractFloat}} 
    optimizer = TDVPl1(A, sampler, TDVPCache(A, params), l1, Ising(), LocalDephasing(), params, ϵ, set_workspace(A, params))
    return optimizer
end

function Initialize!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.optimizer_cache = TDVPCache(optimizer.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.A, optimizer.params)
end

function TDVP_one_body_Lindblad_term!(local_L::T, sample::Projector, j::UInt8, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 

    l1 = optimizer.l1
    A = optimizer.A
    params = optimizer.params
    cache = optimizer.workspace

    s::Matrix{T} = cache.dVEC_transpose[(sample.ket[j],sample.bra[j])]
    mul!(cache.bra_L_l1, s, l1)

    #Iterate over all 4 one-body vectorized basis Projectors:
    @inbounds for (i,state) in zip(1:4,TPSC)
        loc = cache.bra_L_l1[i]
        if loc!=0
            #Compute estimator:
            mul!(cache.loc_1, cache.L_set[j], @view(A[:,:,i]))
            mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
            local_L += loc.*tr(cache.loc_2)
        end
    end
    return local_L
end

function Ising_interaction_energy(ising_op::Ising, sample::Projector, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.A
    params = optimizer.params

    l_int::T=0
    for j::UInt8 in 1:params.N-1
        l_int_ket = (2*sample.ket[j]-1)*(2*sample.ket[j+1]-1)
        l_int_bra = (2*sample.bra[j]-1)*(2*sample.bra[j+1]-1)
        l_int += l_int_ket-l_int_bra
    end
    l_int_ket = (2*sample.ket[params.N]-1)*(2*sample.ket[1]-1)
    l_int_bra = (2*sample.bra[params.N]-1)*(2*sample.bra[1]-1)
    l_int += l_int_ket-l_int_bra
    return -1.0im*params.J*l_int
end

function SweepLindblad!(sample::Projector, ρ_sample::T, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 

    params = optimizer.params
    micro_sample = optimizer.workspace.micro_sample
    micro_sample = Projector(sample)

    temp_local_L::T = 0

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        temp_local_L = TDVP_one_body_Lindblad_term!(temp_local_L, sample, j, optimizer)
    end

    temp_local_L  /= ρ_sample

    return temp_local_L
end

function Update!(optimizer::TDVPl1{T}, sample::Projector) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.

    params=optimizer.params
    A=optimizer.A
    data=optimizer.optimizer_cache
    cache = optimizer.workspace

    local_L = 0
    l_int = 0

    #println("ENTERING")
    ρ_sample::T = MPO(params,sample,A)#tr(cache.R_set[params.N+1])
    #error()
    cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
    cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

#display(sample)
#display(ρ_sample)
#display(cache.Δ)
#error()

    #Sweep lattice:
    local_L = SweepLindblad!(sample, ρ_sample, optimizer)
#display(local_L)
#error()

    #Add in Ising interaction terms:
    #l_int = Ising_interaction_energy(optimizer.ising_op, sample, optimizer)
    #display(l_int)
    #display(local_L)
    #local_L += l_int

    #Update joint ensemble average:
    data.L∂L.+=local_L*conj(cache.Δ) ###conj(cache.Δ)?

    #Update disjoint ensemble average:
    data.ΔLL.+=conj(cache.Δ)
    #data.ΔLL.+=(cache.Δ)*adoint(local_L)

    #Mean local Lindbladian:
    data.mlL += local_L
end

function UpdateSR!(optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}}
    S::Array{T,2} = optimizer.optimizer_cache.S
    avg_G::Vector{T} = optimizer.optimizer_cache.avg_G
    params::Parameters = optimizer.params
    workspace = optimizer.workspace
    
    G::Vector{T} = reshape(workspace.Δ,4*params.χ^2)
    conj_G = conj(G)
    avg_G.+= G
    mul!(workspace.plus_S,conj_G,transpose(G))
    S.+=workspace.plus_S 
end

function Reconfigure!(optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor

    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC
    ϵ = optimizer.ϵ
    params = optimizer.params

    #Compute metric tensor:
    data.S./=N_MC
    data.avg_G./=N_MC
    conj_avg_G = conj(data.avg_G)
    #data.S-=data.avg_G*transpose(conj_avg_G) 

    #Regularize the metric tensor:
    data.S+=ϵ*Matrix{Int}(I, params.χ*params.χ*4, params.χ*params.χ*4)

    #Reconfigure gradient:
    grad::Array{eltype(data.S),3} = (data.L∂L-data.ΔLL)/N_MC
    #grad::Array{eltype(data.S),3} = (data.L∂L)/N_MC
    #println("(data.L∂L-data.ΔLL)/N_MC:")
    #display(grad)
    #error()
    flat_grad::Vector{eltype(data.S)} = reshape(grad,4*params.χ^2)
    flat_grad = inv(data.S)*flat_grad
    #display(inv(data.S))
    #display(flat_grad)
    #error()
    data.∇ = reshape(flat_grad,params.χ,params.χ,4)
end

function Finalize!(optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}}
    N_MC = optimizer.sampler.N_MC
    data = optimizer.optimizer_cache

    data.mlL /= N_MC
    #println("L∂L:")
    #display(data.L∂L)
    #display(data.ΔLL)
    #display(data.mlL)
    data.ΔLL .*= data.mlL
    #println("ΔLL:")
    #display(data.ΔLL)
    #error()
end

function ComputeGradient!(optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}}

    Initialize!(optimizer)
    sample = optimizer.workspace.sample
    #display(sample)
    #error()

    sample = MPO_Metropolis_burn_in(optimizer)

    #display(sample)
    #error()

    for _ in 1:optimizer.sampler.N_MC

        #Generate sample:
        sample, acc = Mono_Metropolis_sweep_left(sample, optimizer)
        #display(sample)
        #error()
        #sample = Projector(Bool[1], Bool[1])
        #display(sample)
        optimizer.optimizer_cache.acceptance += acc/(optimizer.params.N*optimizer.sampler.N_MC)

        #Compute local estimators:
        Update!(optimizer, sample) 

        #Update metric tensor:
        UpdateSR!(optimizer)
    end
end

function Optimize!(optimizer::TDVPl1{T}, δ::Float64) where {T<:Complex{<:AbstractFloat}}

    Finalize!(optimizer)

    #display(optimizer.optimizer_cache.S)
    #display(inv(optimizer.optimizer_cache.S))
    #error()

    #display(optimizer.optimizer_cache.∇)
    #error()
    Reconfigure!(optimizer)

    ∇  = optimizer.optimizer_cache.∇

    #display(∇)
    #error()

    new_A = similar(optimizer.A)
    new_A = optimizer.A + δ*∇
    optimizer.A = new_A
    optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)
end

function MPI_mean!(optimizer::TDVPl1{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    par_cache = optimizer.optimizer_cache

    MPI.Allreduce!(par_cache.L∂L, +, mpi_cache.comm)
    MPI.Allreduce!(par_cache.ΔLL, +, mpi_cache.comm)
    MPI.Allreduce!(par_cache.S, +, mpi_cache.comm)
    MPI.Allreduce!(par_cache.avg_G, +, mpi_cache.comm)
    #MPI.Allreduce!(par_cache.acceptance, +, mpi_cache.comm)

    mlL = [par_cache.mlL]
    MPI.Reduce!(mlL, +, mpi_cache.comm, root=0)

    acceptance = [par_cache.acceptance]
    MPI.Reduce!(acceptance, +, mpi_cache.comm, root=0)

    if mpi_cache.rank == 0
        par_cache.mlL = mlL[1]/mpi_cache.nworkers
        par_cache.L∂L./=mpi_cache.nworkers
        par_cache.ΔLL./=mpi_cache.nworkers
        par_cache.S./=mpi_cache.nworkers
        par_cache.avg_G./=mpi_cache.nworkers
        par_cache.acceptance=acceptance[1]/mpi_cache.nworkers
    end

end