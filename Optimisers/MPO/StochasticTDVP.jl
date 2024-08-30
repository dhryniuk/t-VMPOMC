export TDVP, TensorComputeGradient!, MPI_mean!, Optimize!


function Initialize!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
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

"""
function NormalizeMPO!(norm, params::Parameters, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    ws = optimizer.workspace
    #=
    _MPO = ws.ID
    for i in 1:params.N
        n = mod1(i,params.uc_size)
        _MPO*=(A[n,:,:,1]+A[n,:,:,4])
    end
    trMPO = tr(_MPO)^(1/params.N)
    A./=trMPO
    =#
    norm = maximum(abs.(optimizer.mpo.A))
    A./=norm
    optimizer.mpo.A = A
end
"""

function UpdateSR!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
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

function Reconfigure!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
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

function Finalize!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
    N_MC = optimizer.sampler.N_MC
    data = optimizer.optimizer_cache

    data.mlL /= N_MC
    data.mlL2 /= N_MC
    data.ΔLL .*= data.mlL
end

function Optimize!(optimizer::TDVP{T}, δ::Float64) where {T<:Complex{<:AbstractFloat}}
    A = optimizer.mpo.A

    Finalize!(optimizer)
    Reconfigure!(optimizer)

    ∇  = optimizer.optimizer_cache.∇

    new_A = similar(A)
    new_A = A + δ*∇
    A = new_A
    optimizer.mpo.A = A
    
    NormalizeMPO!(optimizer.params, optimizer)
end


function Optimize!(norm, optimizer::TDVP{T}, δ::Float64) where {T<:Complex{<:AbstractFloat}}
    A = optimizer.mpo.A

    Finalize!(optimizer)
    Reconfigure!(optimizer)

    ∇  = optimizer.optimizer_cache.∇

    #display(∇)
    #sleep(3)

    new_A = similar(A)
    new_A = A + δ*∇
    A = new_A
    optimizer.mpo.A = A

    #display(optimizer.mpo.A)
    #sleep(3)
    
    NormalizeMPO!(norm, optimizer.params, optimizer)

    #display(optimizer.mpo.A)
    #error()
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

function TensorSweepLindblad!(sample::Projector, ρ_sample::T, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    ws = optimizer.workspace
    liouvillian = optimizer.l1

    reduced_density_matrix::MPO{T} = copy(optimizer.mpo)
    @tensor reduced_density_matrix.A[n,a,b,c] := liouvillian[c,d]*reduced_density_matrix.A[n,a,b,d]

    temp_local_L::T = 0
    for j::UInt16 in 1:params.N
        n = mod1(j, params.uc_size)
        mul!(ws.loc_1, ws.L_set[j], @view(reduced_density_matrix.A[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])]]))
        mul!(ws.loc_2, ws.loc_1, ws.R_set[(params.N+1-j)])
        temp_local_L += tr(ws.loc_2)
    end
    temp_local_L /= ρ_sample

    return temp_local_L, 0
end

function TensorSweepLindblad!(sample::Projector, ρ_sample::T, optimizer::TDVPl2{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    ws = optimizer.workspace
    liouvillian = optimizer.l1
    liouvillian_2 = optimizer.l2

    reduced_density_matrix = copy(optimizer.mpo)
    @tensor reduced_density_matrix.A[n,a,b,c] := liouvillian[c,d]*reduced_density_matrix.A[n,a,b,d]

    temp_local_L::T = 0
    for j::UInt16 in 1:params.N
        n = mod1(j, params.uc_size)
        mul!(ws.loc_1, ws.L_set[j], @view(reduced_density_matrix.A[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])]]))
        mul!(ws.loc_2, ws.loc_1, ws.R_set[(params.N+1-j)])
        temp_local_L += tr(ws.loc_2)
    end

    reduced_density_matrices = zeros(T, params.uc_size, params.χ, params.χ, 4, 4)
    reduced_density_matrix = zeros(T, params.χ, params.χ, 4, 4)
    for n in 1:params.uc_size
        A1 = optimizer.mpo.A[n,:,:,:]
        A2 = optimizer.mpo.A[mod1(n+1,params.uc_size),:,:,:]
        @tensor reduced_density_matrix[a,b,d,f] = liouvillian_2[c,e,d,f]*A1[a,u,c]*A2[u,b,e]
        reduced_density_matrices[n,:,:,:,:] = reduced_density_matrix
    end

    for j::UInt16 in 1:params.N-1
        n = mod1(j, params.uc_size)
        mul!(ws.loc_1, ws.L_set[j], @view(reduced_density_matrices[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])],dINDEX[(sample.ket[j+1],sample.bra[j+1])]]))
        mul!(ws.loc_2, ws.loc_1, ws.R_set[(params.N-j)])
        temp_local_L += tr(ws.loc_2)
    end

    ws.loc_1 = ws.L_set[1]
    ws.loc_2 = deepcopy(ws.loc_1)
    for i in 2:params.N-1
        n = mod1(i, params.uc_size)
        mul!(ws.loc_2, ws.loc_1, @view(optimizer.mpo.A[n,:,:,idx(sample,i)]))
        ws.loc_1 = deepcopy(ws.loc_2)
    end
    n = mod1(params.N, params.uc_size)
    mul!(ws.loc_1, @view(reduced_density_matrices[n,:,:,dINDEX[(sample.ket[params.N],sample.bra[params.N])],dINDEX[(sample.ket[1],sample.bra[1])]]), ws.loc_2)
    temp_local_L += tr(ws.loc_1)

    temp_local_L /= ρ_sample

    return temp_local_L, 0
end



function TensorSweepLindblad!(sample::Projector, ρ_sample::T, optimizer::TDVP_H{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    ws = optimizer.workspace
    liouvillian = optimizer.l1
    liouvillian_2 = reshape(optimizer.l2, 4*4,4*4)

    # ONE-BODY TERMS:
    reduced_density_matrix = copy(optimizer.mpo)
    @tensor reduced_density_matrix.A[n,a,b,c] := liouvillian[c,d]*reduced_density_matrix.A[n,a,b,d]
    temp_local_L::T = 0
    for j in 1:params.N
        n = mod1(j, params.uc_size)
        mul!(ws.loc_1, ws.L_set[j], @view(reduced_density_matrix.A[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])]]))
        mul!(ws.loc_2, ws.loc_1, ws.R_set[(params.N+1-j)])
        temp_local_L += tr(ws.loc_2)
    end

    # INTRA-LAYER TERMS:
    temp_local_L_int::T = 0
    for i in 1:params.uc_size
        for j in 1:params.uc_size-1
            k = j+(i-1)*params.uc_size

            range = circshift(Array(1:params.N),-k)
            n = mod1(k, params.uc_size)
            m = mod1(k+1, params.uc_size)

            B1 = copy(ws.L_set[1])
            B2 = ws.R_set[(params.N-k)] * ws.L_set[k]

            s1 = ws.dVEC_transpose[(sample.ket[k],sample.bra[k])]
            s2 = ws.dVEC_transpose[(sample.ket[k+1],sample.bra[k+1])]
            ws.s = kron(s1,s2)
            mul!(ws.bra_L_l2, ws.s, liouvillian_2)

            explicit_summation = 0
            for (u, state_u) in zip(1:4, TLS_Liouville_Space)
                for (v, state_v) in zip(1:4, TLS_Liouville_Space)
                    loc = ws.bra_L_l2[v+4*(u-1)]
                    if loc!=0
                        reduced_density_matrix = loc * optimizer.mpo.A[n,:,:,u] * optimizer.mpo.A[m,:,:,v] * B2
                        explicit_summation += tr(reduced_density_matrix)
                    end
                end
            end

            temp_local_L_int += explicit_summation
        end
    end

    # BOUNDARY-LAYER TERMS:
    for i in 1:params.uc_size
        j = 1+(i-1)*params.uc_size
        k = i*params.uc_size

        range = circshift(Array(1:params.N),-k)[1:end-params.uc_size]
        n = mod1(j, params.uc_size)
        m = mod1(k, params.uc_size)

        B1 = copy(ws.L_set[1])
        B2 = copy(ws.L_set[1])

        for l in j+1:k-1
            p = mod1(l, params.uc_size)
            B1 = B1*optimizer.mpo.A[p,:,:,dINDEX[(sample.ket[l],sample.bra[l])]]
        end
        B2 = ws.R_set[(params.N+1-k)] * ws.L_set[j]

        s1 = ws.dVEC_transpose[(sample.ket[j],sample.bra[j])]
        s2 = ws.dVEC_transpose[(sample.ket[k],sample.bra[k])]
        ws.s = kron(s1,s2)
        mul!(ws.bra_L_l2, ws.s, liouvillian_2)

        explicit_summation = 0
        for (u, state_u) in zip(1:4, TLS_Liouville_Space)
            for (v, state_v) in zip(1:4, TLS_Liouville_Space)
                loc = ws.bra_L_l2[v+4*(u-1)]
                if loc!=0
                    reduced_density_matrix = loc * optimizer.mpo.A[n,:,:,u] * B1 * optimizer.mpo.A[m,:,:,v] * B2
                    explicit_summation += tr(reduced_density_matrix)
                end
            end
        end

        temp_local_L_int += explicit_summation
    end

    # CROSS-LAYER TERMS:
    for j in 1:params.N
        range = circshift(Array(1:params.N),-j)
        n = mod1(j, params.uc_size)
        k = mod1(j+params.uc_size, params.N)


        B1 = copy(ws.L_set[1])
        B2 = copy(ws.L_set[1])

        for l in range[1:params.uc_size-1]
            m = mod1(l, params.uc_size)
            B1 = B1*optimizer.mpo.A[m,:,:,dINDEX[(sample.ket[l],sample.bra[l])]]
        end
        if k>j
            B2 = ws.R_set[(params.N+1-k)] * ws.L_set[j] 
        else
            for l in range[params.uc_size+1:end-1]
                m = mod1(l, params.uc_size)
                B2 = B2*optimizer.mpo.A[m,:,:,dINDEX[(sample.ket[l],sample.bra[l])]]
            end
        end

        s1 = ws.dVEC_transpose[(sample.ket[j],sample.bra[j])]
        s2 = ws.dVEC_transpose[(sample.ket[k],sample.bra[k])]
        ws.s = kron(s1,s2)
        mul!(ws.bra_L_l2, ws.s, liouvillian_2)

        explicit_summation = 0
        for (u, state_u) in zip(1:4, TLS_Liouville_Space)
            for (v, state_v) in zip(1:4, TLS_Liouville_Space)
                loc = ws.bra_L_l2[v+4*(u-1)]
                if loc!=0
                    reduced_density_matrix = loc * optimizer.mpo.A[n,:,:,u] * B1 * optimizer.mpo.A[n,:,:,v] * B2
                    explicit_summation += tr(reduced_density_matrix)
                end
            end
        end
    
        temp_local_L_int += explicit_summation
    end

    temp_local_L /= ρ_sample
    temp_local_L_int /= ρ_sample

    return temp_local_L, temp_local_L_int
end

function TensorUpdate!(optimizer::TDVP{T}, sample::Projector) where {T<:Complex{<:AbstractFloat}}
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

    #local_L += local_L_int
    local_L += l_int

    #Update joint ensemble average:
    data.L∂L.+=local_L*conj(ws.Δ)

    #Update disjoint ensemble average:
    data.ΔLL.+=conj(ws.Δ)

    #Mean local Lindbladian:
    data.mlL += local_L
    data.mlL2 += abs2(local_L)
    #data.mlL += local_L_int
    #data.mlL2 += l_int

    #return l_int
end

function TensorComputeGradient!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
    Initialize!(optimizer)
    sample = optimizer.workspace.sample
    sample = MPO_Metropolis_burn_in!(optimizer)

    interaction_energy = 0.0+0.0im

    for _ in 1:optimizer.sampler.N_MC

        #Generate sample:
        sample, acc = MetropolisSweepLeft!(sample, optimizer)
        optimizer.optimizer_cache.acceptance += acc/(optimizer.params.N*optimizer.sampler.N_MC)

        #Compute local estimators:
        TensorUpdate!(optimizer, sample) 

        #Update metric tensor:
        UpdateSR!(optimizer)
    end
end