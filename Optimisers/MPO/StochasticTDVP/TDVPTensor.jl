#export TDVP, TensorComputeGradient!, MPI_mean!, Optimize!


function Initialize!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.sampler = MetropolisSampler(optimizer.sampler.N_MC, optimizer.sampler.N_MC_Heun, optimizer.sampler.burn, optimizer.params) # resets samples lists!
    optimizer.optimizer_cache = TDVPCache(optimizer.mpo.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.mpo.A, optimizer.params)
end

function NormalizeMPO!(params::Parameters, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    ws = optimizer.workspace
    
    _MPO = ws.ID
    for i in 1:params.N
        n = mod1(i,params.uc_size)
        _MPO*=(A[n,:,:,1]+A[n,:,:,4])
    end
    trMPO = tr(_MPO)^(1/params.N)
    A./=trMPO
    optimizer.mpo.A = A
end

function UpdateSR!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
    S::Array{T,2} = optimizer.optimizer_cache.S
    avg_G::Vector{T} = optimizer.optimizer_cache.avg_G
    params::Parameters = optimizer.params
    ws = optimizer.workspace
    
    G::Vector{T} = reshape(ws.Δ,params.uc_size*4*params.χ^2)
    #conj_G = conj(G)
    avg_G .+= G
    #BLAS.herk!('U', 'N', 1.0, transpose(G), 1.0, ws.plus_S) 
    mul!(ws.plus_S,conj.(G),transpose(G)) ### VERY EXPENSIVE
    BLAS.axpy!(1.0, ws.plus_S, S)
    #@inbounds S .+= ws.plus_S ### MOST EXPENSIVE
end


function Finalize!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
    N_MC = optimizer.sampler.N_MC
    data = optimizer.optimizer_cache

    data.mlL /= N_MC
    data.mlL2 /= N_MC
    data.ΔLL .*= data.mlL
end

function Optimize!(optimizer::TDVP{T}, δ::Float64, estimators, gradients) where {T<:Complex{<:AbstractFloat}}
    A = optimizer.mpo.A

    Finalize!(optimizer)
#    Reconfigure!(optimizer, optimizer.sampler.estimators, optimizer.sampler.gradients)
    Reconfigure!(optimizer, estimators, gradients)

    ∇  = optimizer.optimizer_cache.∇

    new_A = similar(A)
    new_A = A + δ*∇
    A = new_A
    optimizer.mpo.A = A
    
    NormalizeMPO!(optimizer.params, optimizer)
end

function MPI_mean!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
    par_cache = optimizer.optimizer_cache
    comm = mpi_cache.comm
    rank = mpi_cache.rank
    nworkers = mpi_cache.nworkers
    params = optimizer.params

    MPI.Allreduce!(par_cache.L∂L, +, comm)
    MPI.Allreduce!(par_cache.ΔLL, +, comm)
    MPI.Allreduce!(par_cache.S, +, comm)
    MPI.Allreduce!(par_cache.avg_G, +, comm)

    mlL = [par_cache.mlL]
    MPI.Reduce!(mlL, +, comm, root=0)

    acceptance = [par_cache.acceptance]
    MPI.Reduce!(acceptance, +, comm, root=0)

    if mpi_cache.rank == 0
        par_cache.mlL = mlL[1]/nworkers
        par_cache.L∂L./=nworkers
        par_cache.ΔLL./=nworkers
        par_cache.S./=nworkers
        par_cache.avg_G./=nworkers
        par_cache.acceptance=acceptance[1]/nworkers
    end


    # Gather sampled local estimators and gradients:

    loc_estimators = optimizer.sampler.estimators 
    if rank == 0
        concat_estimators = Vector{T}(undef, optimizer.sampler.N_MC * nworkers)
    else
        concat_estimators = Vector{T}(undef, 0)
    end
    MPI.Gather!(loc_estimators, concat_estimators, 0, comm)

    loc_gradients = optimizer.sampler.gradients 
    if rank == 0
        #concat_gradients = zeros(T, optimizer.sampler.N_MC* nworkers, params.uc_size, params.χ, params.χ, 4)
        concat_gradients = zeros(T, nworkers*optimizer.sampler.N_MC, params.uc_size*params.χ*params.χ*4)
    else
        concat_gradients = Vector{T}(undef, 0)
    end
    MPI.Gather!(loc_gradients, concat_gradients, 0, comm)

    return concat_estimators, concat_gradients
end

#function TensorUpdate!(optimizer::TDVP{T}, sample::Projector) where {T<:Complex{<:AbstractFloat}}
#function TensorUpdate!(optimizer::TDVP{T}, sample::Projector, local_estimators, gradients) where {T<:Complex{<:AbstractFloat}}
function TensorUpdate!(optimizer::TDVP{T}, sample::Projector, n::Int64) where {T<:Complex{<:AbstractFloat}}
    params=optimizer.params
    mpo=optimizer.mpo
    data=optimizer.optimizer_cache
    ws = optimizer.workspace

    local_L::T = 0
    local_L_int::T = 0
    l_int::T = 0

    ρ_sample::T = tr(ws.R_set[params.N+1])
    ws.L_set = L_MPO_products!(ws.L_set, sample, mpo, params, ws)
    ws.Δ = ∂MPO(sample, ws.L_set, ws.R_set, params, ws, optimizer.mpo)./ρ_sample

    #Sweep lattice:
    local_L, local_L_int = TensorSweepLindblad!(sample, ρ_sample, optimizer)

    #Add in Ising interaction terms:
    l_int = IsingInteractionEnergy(optimizer.ising_op, sample, optimizer)

    local_L += local_L_int
    local_L += l_int

    #Update joint ensemble average:
    data.L∂L.+=local_L*conj(ws.Δ)

    #Update disjoint ensemble average:
    data.ΔLL.+=conj(ws.Δ)

    #Mean local Lindbladian:
    data.mlL += local_L
    data.mlL2 += abs2(local_L)

    optimizer.sampler.estimators[n] = copy(local_L)
    #optimizer.sampler.gradients[n,:,:,:,:] = deepcopy(ws.Δ)
    optimizer.sampler.gradients[n,:] = reshape(ws.Δ,4*params.χ^2*params.uc_size)
end

function TensorComputeGradient!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
#function TensorComputeGradient!(optimizer::TDVP{T}, local_estimators, gradients) where {T<:Complex{<:AbstractFloat}}
    Initialize!(optimizer)
    sample = optimizer.workspace.sample
    sample = MPO_Metropolis_burn_in!(optimizer)

    for n in 1:optimizer.sampler.N_MC

        #Generate sample:
        sample, acc = MetropolisSweepLeft!(sample, 5, optimizer)
        optimizer.optimizer_cache.acceptance += acc/(optimizer.params.N*optimizer.sampler.N_MC)

        #Compute local estimators:
        #TensorUpdate!(optimizer, sample) 
        TensorUpdate!(optimizer, sample, n)#local_estimators, gradients) 

        #Update metric tensor:
        UpdateSR!(optimizer)
    end
end
