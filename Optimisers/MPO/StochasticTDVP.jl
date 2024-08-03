export TDVP, compute_gradient!, tensor_compute_gradient!, MPI_mean!, optimize!#, optimize_no_normalize!

function initialize!(optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.optimizer_cache = TDVPCache(optimizer.mpo.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.mpo.A, optimizer.params)
end

function initialize!(optimizer::PTI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.optimizer_cache = PTI_TDVPCache(optimizer.mpo.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.mpo.A, optimizer.params)
end

function TDVP_one_body_Lindblad_term!(local_L::T, sample::Projector, j::UInt16, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    l1 = optimizer.list_l1[j]
    A = optimizer.mpo.A
    params = optimizer.params
    cache = optimizer.workspace

    s::Matrix{T} = cache.dVEC_transpose[(sample.ket[j],sample.bra[j])]
    mul!(cache.bra_L_l1, s, l1)

    #Iterate over all 4 one-body vectorized basis Projectors:
    @inbounds for (i,state) in zip(1:4,TPSC)
        loc = cache.bra_L_l1[i]
        if loc!=0
            #Compute estimator:
            mul!(cache.loc_1, cache.L_set[j], @view(A[j,:,:,i]))
            mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
            local_L += loc.*tr(cache.loc_2)
        end
    end
    return local_L
end

function normalize_MPO!(params::Parameters, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    cache = optimizer.workspace
    
    _MPO = cache.ID
    for i in 1:params.N
        _MPO*=(A[i,:,:,1]+A[i,:,:,4])
    end
    trMPO = tr(_MPO)^(1/params.N)
    A./=trMPO
    optimizer.mpo.A = A
end

function normalize_MPO!(params::Parameters, optimizer::TI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    _MPO=(A[:,:,1]+A[:,:,4])^params.N
    A./=tr(_MPO)^(1/params.N)
    optimizer.mpo.A = A
end

function normalize_MPO!(params::Parameters, optimizer::PTI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    cache = optimizer.workspace
    
    _MPO = cache.ID
    for i in 1:params.N
        n = mod1(i,params.uc_size)
        _MPO*=(A[n,:,:,1]+A[n,:,:,4])
    end
    trMPO = tr(_MPO)^(1/params.N)
    A./=trMPO
    optimizer.mpo.A = A
end

function initialize!(optimizer::TI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.optimizer_cache = TI_TDVPCache(optimizer.mpo.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.mpo.A, optimizer.params)
end

function TDVP_one_body_Lindblad_term!(local_L::T, sample::Projector, j::UInt16, optimizer::TI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    l1 = optimizer.l1
    A = optimizer.mpo.A
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

function Ising_interaction_energy(ising_op::Ising, sample::Projector, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params

    l_int::T=0
    for j::UInt16 in 1:params.N-1
        l_int_ket = (2*sample.ket[j]-1)*(2*sample.ket[j+1]-1)
        l_int_bra = (2*sample.bra[j]-1)*(2*sample.bra[j+1]-1)
        l_int += l_int_ket-l_int_bra
    end
    l_int_ket = (2*sample.ket[params.N]-1)*(2*sample.ket[1]-1)
    l_int_bra = (2*sample.bra[params.N]-1)*(2*sample.bra[1]-1)
    l_int += l_int_ket-l_int_bra

    return -1.0im*params.J*l_int
end

function Ising_interaction_energy(ising_op::LongRangeIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params

    l_int_ket::T = 0.0
    l_int_bra::T = 0.0
    l_int::T = 0.0
    for i::Int16 in 1:params.N-1
        for j::Int16 in i+1:params.N
            l_int_ket = (2*sample.ket[i]-1)*(2*sample.ket[j]-1)
            l_int_bra = (2*sample.bra[i]-1)*(2*sample.bra[j]-1)
            dist = min(abs(i-j), abs(params.N+i-j))^ising_op.α
            l_int += (l_int_ket-l_int_bra)/dist
        end
    end
    return -1.0im*params.J*l_int/ising_op.Kac_norm
end

function Ising_interaction_energy(ising_op::SquareIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params
    L = isqrt(params.N)

    l_int::T=0
    for k::UInt16 in 0:L-1
        for j::UInt16 in 1:L

            #Horizontal:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[mod(j,L)+1+k*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[mod(j,L)+1+k*L]-1)
            l_int += l_int_ket-l_int_bra

            #Vertical:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[j+mod(k+1,L)*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[j+mod(k+1,L)*L]-1)
            l_int += l_int_ket-l_int_bra

        end
    end

    return -1.0im*params.J*l_int
end

function Ising_interaction_energy(ising_op::TriangularIsing, sample::Projector, optimizer::Optimizer{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params
    L = isqrt(params.N)

    l_int::T=0
    for k::UInt16 in 0:L-1
        for j::UInt16 in 1:L

            #Horizontal:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[mod(j,L)+1+k*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[mod(j,L)+1+k*L]-1)
            l_int += l_int_ket-l_int_bra

            #Vertical:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[j+mod(k+1,L)*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[j+mod(k+1,L)*L]-1)
            l_int += l_int_ket-l_int_bra

            #Diagonal:
            l_int_ket = (2*sample.ket[j+k*L]-1)*(2*sample.ket[mod(j,L)+1+mod(k+1,L)*L]-1)
            l_int_bra = (2*sample.bra[j+k*L]-1)*(2*sample.bra[mod(j,L)+1+mod(k+1,L)*L]-1)
            l_int += l_int_ket-l_int_bra

        end
    end

    return -1.0im*params.J*l_int
end

function sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    micro_sample = optimizer.workspace.micro_sample
    micro_sample = Projector(sample)

    temp_local_L::T = 0

    #Calculate L∂L*:
    for j::UInt16 in 1:params.N
        temp_local_L = TDVP_one_body_Lindblad_term!(temp_local_L, sample, j, optimizer)
    end

    temp_local_L  /= ρ_sample

    return temp_local_L
end

function update!(optimizer::TDVP{T}, sample::Projector) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.
    params=optimizer.params
    mpo=optimizer.mpo
    data=optimizer.optimizer_cache
    cache = optimizer.workspace

    local_L = 0
    l_int = 0

    #ρ_sample::T = trMPO(params, sample, mpo)  ### CAN THIS BE SIMPLIFIED BY REUSING PARTIAL MATRIX PRODUCTS?
    ρ_sample::T = tr(cache.R_set[params.N+1])
    cache.L_set = L_MPO_products!(cache.L_set, sample, mpo, params, cache)
    cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache, optimizer.mpo)./ρ_sample

    #Sweep lattice:
    local_L = sweep_Lindblad!(sample, ρ_sample, optimizer)

    #Add in Ising interaction terms:
    l_int = Ising_interaction_energy(optimizer.ising_op, sample, optimizer)
    local_L += l_int

    #Update joint ensemble average:
    data.L∂L.+=local_L*conj(cache.Δ)

    #Update disjoint ensemble average:
    data.ΔLL.+=conj(cache.Δ)
    #data.ΔLL.+=(cache.Δ)*adoint(local_L)

    #Mean local Lindbladian:
    data.mlL += local_L
end

function update_SR!(optimizer::TI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}}
    S::Array{T,2} = optimizer.optimizer_cache.S
    avg_G::Vector{T} = optimizer.optimizer_cache.avg_G
    params::Parameters = optimizer.params
    ws = optimizer.workspace
    
    G::Vector{T} = reshape(ws.Δ,4*params.χ^2)
    conj_G = conj(G)
    avg_G .+= G
    mul!(ws.plus_S,conj_G,transpose(G))
    S .+= ws.plus_S 

    #G::Vector{T} = reshape(workspace.Δ,prod(size(workspace.Δ))) # prod(size(Δ))
    #conj_G = conj(G)
    #avg_G.+= G
    #mul!(workspace.plus_S,conj_G,transpose(G))
    #S.+=workspace.plus_S ###SLOWEST PART BY FAR
end

function update_SR!(optimizer::PTI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}}
    S::Array{T,2} = optimizer.optimizer_cache.S
    avg_G::Vector{T} = optimizer.optimizer_cache.avg_G
    params::Parameters = optimizer.params
    ws = optimizer.workspace
    
    G::Vector{T} = reshape(ws.Δ,params.uc_size*4*params.χ^2)
    conj_G = conj(G)
    avg_G .+= G
    mul!(ws.plus_S,conj_G,transpose(G))
    S .+= ws.plus_S 
end



function reconfigure!(optimizer::TI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC
    ϵ = optimizer.ϵ
    params = optimizer.params

    #Compute metric tensor:
    data.S./=N_MC
    data.avg_G./=N_MC
    conj_avg_G = conj(data.avg_G)
    data.S-=data.avg_G*transpose(conj_avg_G)  ### WARNING: MAY NOT BE CORRECT !!!

    #Regularize the metric tensor:
    data.S+=ϵ*Matrix{Int}(I, size(data.S))

    #Reconfigure gradient:
    grad = (data.L∂L-data.ΔLL)/N_MC
    flat_grad = reshape(grad,prod(size(grad)))
    flat_grad = inv(data.S)*flat_grad

    ### NEED TO CONVERT ARRAY INTO VECTOR(ARRAY)

    data.∇ = reshape(flat_grad,size(data.∇))
end
"""
function reconfigure!(optimizer::TI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC
    ϵ = optimizer.ϵ
    params = optimizer.params

    #Compute metric tensor:
    rmul!(data.S, 1/N_MC)
    rmul!(data.avg_G, 1/N_MC)
    conj_avg_G = conj(data.avg_G)
    data.S -= data.avg_G*transpose(conj_avg_G)

    # Regularize the metric tensor
    @inbounds for i in 1:size(data.S, 1)
        data.S[i,i] += ϵ
    end
    
    # Reconfigure gradient
    grad_size = (params.χ, params.χ, 4)
    flat_grad_size = 4*params.χ^2
    
    grad = reshape(view(data.L∂L, :), grad_size) .- reshape(view(data.ΔLL, :), grad_size)
    rmul!(grad, 1/N_MC)
    flat_grad = reshape(grad, flat_grad_size)
    
    flat_grad = inv(data.S)*flat_grad
    data.∇ = reshape(flat_grad, params.χ, params.χ, 4)
end
"""

function reconfigure!(optimizer::PTI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC
    ϵ = optimizer.ϵ
    params = optimizer.params

    #Compute metric tensor:
    data.S./=N_MC
    data.avg_G./=N_MC
    conj_avg_G = conj(data.avg_G)
    data.S-=data.avg_G*transpose(conj_avg_G)  ### WARNING: MAY NOT BE CORRECT !!!

    #Regularize the metric tensor:
    data.S+=ϵ*Matrix{Int}(I, size(data.S))

    #Reconfigure gradient:
    grad = (data.L∂L-data.ΔLL)/N_MC
    flat_grad = reshape(grad,prod(size(grad)))
    flat_grad = inv(data.S)*flat_grad

    ### NEED TO CONVERT ARRAY INTO VECTOR(ARRAY)

    data.∇ = reshape(flat_grad,size(data.∇))
end

function finalize!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
    N_MC = optimizer.sampler.N_MC
    data = optimizer.optimizer_cache

    data.mlL /= N_MC
    data.ΔLL .*= data.mlL
end

function compute_gradient!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}

    initialize!(optimizer)
    sample = optimizer.workspace.sample

    sample = MPO_Metropolis_burn_in!(optimizer)

    for _ in 1:optimizer.sampler.N_MC

        #Generate sample:
        sample, acc = Metropolis_sweep_left!(sample, optimizer)
        optimizer.optimizer_cache.acceptance += acc/(optimizer.params.N*optimizer.sampler.N_MC)

        #Compute local estimators:
        update!(optimizer, sample) 

        #Update metric tensor:
        update_SR!(optimizer)
    end
end

function optimize!(optimizer::TDVP{T}, δ::Float64) where {T<:Complex{<:AbstractFloat}}
    A = optimizer.mpo.A

    finalize!(optimizer)
    reconfigure!(optimizer)

    ∇  = optimizer.optimizer_cache.∇

    new_A = similar(A)
    new_A = A + δ*∇
    A = new_A
    optimizer.mpo.A = A
    
    normalize_MPO!(optimizer.params, optimizer)
end

function MPI_mean!(optimizer::TDVP{T}, mpi_cache) where {T<:Complex{<:AbstractFloat}}
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

function tensor_sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    cache = optimizer.workspace
    liouvillian = optimizer.l1

    reduced_density_matrix::TI_MPO{T} = copy(optimizer.mpo)
    @tensor reduced_density_matrix.A[a,b,c] := liouvillian[c,d]*reduced_density_matrix.A[a,b,d]

    temp_local_L::T = 0
    for j::UInt16 in 1:params.N
        mul!(cache.loc_1, cache.L_set[j], @view(reduced_density_matrix.A[:,:,dINDEX[(sample.ket[j],sample.bra[j])]]))
        mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
        temp_local_L += tr(cache.loc_2)
    end
    temp_local_L /= ρ_sample

    return temp_local_L
end

function tensor_sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::PTI_TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    cache = optimizer.workspace
    liouvillian = optimizer.l1

    reduced_density_matrix::PTI_MPO{T} = copy(optimizer.mpo)
    @tensor reduced_density_matrix.A[n,a,b,c] := liouvillian[c,d]*reduced_density_matrix.A[n,a,b,d]

    temp_local_L::T = 0
    for j::UInt16 in 1:params.N
        n = mod1(j, params.uc_size)
        mul!(cache.loc_1, cache.L_set[j], @view(reduced_density_matrix.A[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])]]))
        mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
        temp_local_L += tr(cache.loc_2)
    end
    temp_local_L /= ρ_sample

    return temp_local_L
end



function tensor_update!(optimizer::TDVP{T}, sample::Projector, interaction_energy) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.
    params=optimizer.params
    mpo=optimizer.mpo
    data=optimizer.optimizer_cache
    cache = optimizer.workspace

    local_L::T = 0
    l_int::T = 0

    #ρ_sample::T = trMPO(params, sample, mpo)  ### CAN THIS BE SIMPLIFIED BY REUSING PARTIAL MATRIX PRODUCTS?
    ρ_sample::T = tr(cache.R_set[params.N+1])
    cache.L_set = L_MPO_products!(cache.L_set, sample, mpo, params, cache)
    cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache, optimizer.mpo)./ρ_sample

    #Sweep lattice:
    local_L = tensor_sweep_Lindblad!(sample, ρ_sample, optimizer)

    #Add in Ising interaction terms:
    l_int = Ising_interaction_energy(optimizer.ising_op, sample, optimizer)
    local_L += l_int

    #Update joint ensemble average:
    data.L∂L.+=local_L*conj(cache.Δ)

    #Update disjoint ensemble average:
    data.ΔLL.+=conj(cache.Δ)

    #Mean local Lindbladian:
    data.mlL += local_L

    return l_int
end

function tensor_compute_gradient!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
    initialize!(optimizer)
    sample = optimizer.workspace.sample
    sample = MPO_Metropolis_burn_in!(optimizer)

    interaction_energy = 0.0+0.0im

    for _ in 1:optimizer.sampler.N_MC

        #Generate sample:
        sample, acc = Metropolis_sweep_left!(sample, optimizer)
        optimizer.optimizer_cache.acceptance += acc/(optimizer.params.N*optimizer.sampler.N_MC)

        #Compute local estimators:
        interaction_energy += tensor_update!(optimizer, sample, interaction_energy) 

        #Update metric tensor:
        update_SR!(optimizer)
    end
end