export TDVP, tensor_compute_gradient!, MPI_mean!, optimize!


function initialize!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.optimizer_cache = TDVPCache(optimizer.mpo.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.mpo.A, optimizer.params)
end

function normalize_MPO!(params::Parameters, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
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

function update_SR!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}}
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

function reconfigure!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
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

    #display(data.ΔLL)
    #display(data.mlL)
    #error()

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
    data.mlL2 /= N_MC
    data.ΔLL .*= data.mlL
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

function tensor_sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    cache = optimizer.workspace
    liouvillian = optimizer.l1

    reduced_density_matrix::MPO{T} = copy(optimizer.mpo)
    @tensor reduced_density_matrix.A[n,a,b,c] := liouvillian[c,d]*reduced_density_matrix.A[n,a,b,d]

    temp_local_L::T = 0
    for j::UInt16 in 1:params.N
        n = mod1(j, params.uc_size)
        mul!(cache.loc_1, cache.L_set[j], @view(reduced_density_matrix.A[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])]]))
        mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
        temp_local_L += tr(cache.loc_2)
    end
    temp_local_L /= ρ_sample

    return temp_local_L, 0
end

function tensor_sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::TDVPl2{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    cache = optimizer.workspace
    liouvillian = optimizer.l1
    liouvillian_2 = optimizer.l2

    reduced_density_matrix = copy(optimizer.mpo)
    @tensor reduced_density_matrix.A[n,a,b,c] := liouvillian[c,d]*reduced_density_matrix.A[n,a,b,d]

    temp_local_L::T = 0
    for j::UInt16 in 1:params.N
        n = mod1(j, params.uc_size)
        mul!(cache.loc_1, cache.L_set[j], @view(reduced_density_matrix.A[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])]]))
        mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
        temp_local_L += tr(cache.loc_2)
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
        mul!(cache.loc_1, cache.L_set[j], @view(reduced_density_matrices[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])],dINDEX[(sample.ket[j+1],sample.bra[j+1])]]))
        mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N-j)])
        temp_local_L += tr(cache.loc_2)
    end

    cache.loc_1 = cache.L_set[1]
    cache.loc_2 = deepcopy(cache.loc_1)
    for i in 2:params.N-1
        n = mod1(i, params.uc_size)
        mul!(cache.loc_2, cache.loc_1, @view(optimizer.mpo.A[n,:,:,idx(sample,i)]))
        cache.loc_1 = deepcopy(cache.loc_2)
    end
    n = mod1(params.N, params.uc_size)
    mul!(cache.loc_1, @view(reduced_density_matrices[n,:,:,dINDEX[(sample.ket[params.N],sample.bra[params.N])],dINDEX[(sample.ket[1],sample.bra[1])]]), cache.loc_2)
    temp_local_L += tr(cache.loc_1)

    temp_local_L /= ρ_sample

    return temp_local_L, 0
end



function tensor_sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::TDVP_H{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    cache = optimizer.workspace
    liouvillian = optimizer.l1
    liouvillian_2 = reshape(optimizer.l2, 4*4,4*4)

    # ONE-BODY TERMS:
    reduced_density_matrix = copy(optimizer.mpo)
    @tensor reduced_density_matrix.A[n,a,b,c] := liouvillian[c,d]*reduced_density_matrix.A[n,a,b,d]
    temp_local_L::T = 0
    for j in 1:params.N
        n = mod1(j, params.uc_size)
        mul!(cache.loc_1, cache.L_set[j], @view(reduced_density_matrix.A[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])]]))
        mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
        temp_local_L += tr(cache.loc_2)
    end

    """
    # INTRA-LAYER TERMS:
    reduced_density_matrices = zeros(T, params.uc_size-1, params.χ, params.χ, 4, 4)
    reduced_density_matrix = zeros(T, params.χ, params.χ, 4, 4)
    temp_local_L_int::T = 0
    for n in 1:params.uc_size-1
        A1 = optimizer.mpo.A[n,:,:,:]
        A2 = optimizer.mpo.A[n+1,:,:,:]
#        @tensor reduced_density_matrix[a,b,d,f] = optimizer.l2[c,e,d,f]*A1[a,u,c]*A2[u,b,e]
@tensor reduced_density_matrix[a,b,d,f] = optimizer.l2[c,d,e,f]*A1[a,u,c]*A2[u,b,e]
        reduced_density_matrices[n,:,:,:,:] = reduced_density_matrix
        display(optimizer.l2)
        display(reduced_density_matrix)
        error()
    end
    display(ρ_sample)
    for j in 1:params.N-1
        if mod(j,params.uc_size)!=0
            n = mod1(j, params.uc_size)
            mul!(cache.loc_1, cache.L_set[j], @view(reduced_density_matrices[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])],dINDEX[(sample.ket[j+1],sample.bra[j+1])]]))
            mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N-j)])
            temp_local_L_int += tr(cache.loc_2)
            println(j,";",j+1)
            println(tr(cache.loc_2)/ρ_sample)
        end
    end
    """

    #display(optimizer.mpo.A)

    # INTRA-LAYER TERMS:
    temp_local_L_int::T = 0
    for i in 1:params.uc_size
        for j in 1:params.uc_size-1
            k = j+(i-1)*params.uc_size

            #println(k,";",k+1)
            
            range = circshift(Array(1:params.N),-k)#[1:end-params.uc_size]
            n = mod1(k, params.uc_size)
            m = mod1(k+1, params.uc_size)

            B1 = copy(cache.L_set[1])
            B2 = cache.R_set[(params.N-k)] * cache.L_set[k]

            #display(cache.L_set[k])
            #display(cache.R_set[(params.N-k)])

            #display(B2)


            s1 = cache.dVEC_transpose[(sample.ket[k],sample.bra[k])]
            s2 = cache.dVEC_transpose[(sample.ket[k+1],sample.bra[k+1])]
            cache.s = kron(s1,s2)
            mul!(cache.bra_L_l2, cache.s, liouvillian_2)

            #display(cache.bra_L_l2)


            explicit_summation = 0
            for (u, state_u) in zip(1:4, TLS_Liouville_Space)
                for (v, state_v) in zip(1:4, TLS_Liouville_Space)
                    loc = cache.bra_L_l2[v+4*(u-1)]
                    if loc!=0
                        #display(loc)
                        #display(state_u)
                        #display(state_v)
                        #error()
                        reduced_density_matrix = loc * optimizer.mpo.A[n,:,:,u] * optimizer.mpo.A[m,:,:,v] * B2
                        explicit_summation += tr(reduced_density_matrix)
                    end
                end
            end

            temp_local_L_int += explicit_summation

            #println(explicit_summation/ρ_sample)
        end
    end

    # BOUNDARY-LAYER TERMS:
    for i in 1:params.uc_size
        j = 1+(i-1)*params.uc_size
        k = i*params.uc_size

        #println(j,";",k)
        
        range = circshift(Array(1:params.N),-k)[1:end-params.uc_size]
        n = mod1(j, params.uc_size)
        m = mod1(k, params.uc_size)

        B1 = copy(cache.L_set[1])
        B2 = copy(cache.L_set[1])

        #display(B1)
        #error()

        for l in j+1:k-1
            p = mod1(l, params.uc_size)
            B1 = B1*optimizer.mpo.A[p,:,:,dINDEX[(sample.ket[l],sample.bra[l])]]
        end
        B2 = cache.R_set[(params.N+1-k)] * cache.L_set[j]

        s1 = cache.dVEC_transpose[(sample.ket[j],sample.bra[j])]
        s2 = cache.dVEC_transpose[(sample.ket[k],sample.bra[k])]
        cache.s = kron(s1,s2)
        mul!(cache.bra_L_l2, cache.s, liouvillian_2)

#        display(sample)
#        display(cache.bra_L_l2)
#        error()
        explicit_summation = 0
        for (u, state_u) in zip(1:4, TLS_Liouville_Space)
            for (v, state_v) in zip(1:4, TLS_Liouville_Space)
                loc = cache.bra_L_l2[v+4*(u-1)]
                if loc!=0
                    reduced_density_matrix = loc * optimizer.mpo.A[n,:,:,u] * B1 * optimizer.mpo.A[m,:,:,v] * B2
                    explicit_summation += tr(reduced_density_matrix)
                end
            end
        end

        temp_local_L_int += explicit_summation

        #println(explicit_summation/ρ_sample)
    end

    # CROSS-LAYER TERMS:
    for j in 1:params.N
        range = circshift(Array(1:params.N),-j)
        n = mod1(j, params.uc_size)
        k = mod1(j+params.uc_size, params.N)

        #println(j,";",k)

        B1 = copy(cache.L_set[1])
        B2 = copy(cache.L_set[1])

        for l in range[1:params.uc_size-1]
            m = mod1(l, params.uc_size)
            B1 = B1*optimizer.mpo.A[m,:,:,dINDEX[(sample.ket[l],sample.bra[l])]]
        end
        #B2 = cache.R_set[(params.N+1-k)] * cache.L_set[j]   #####  !!! H E R E !!!
        if k>j
            B2 = cache.R_set[(params.N+1-k)] * cache.L_set[j] 
        else
            for l in range[params.uc_size+1:end-1]
                m = mod1(l, params.uc_size)
                B2 = B2*optimizer.mpo.A[m,:,:,dINDEX[(sample.ket[l],sample.bra[l])]]
            end
        end

        #println("B1: ", B1)
        #println("B2: ", B2)

        s1 = cache.dVEC_transpose[(sample.ket[j],sample.bra[j])]
        s2 = cache.dVEC_transpose[(sample.ket[k],sample.bra[k])]
        cache.s = kron(s1,s2)
        mul!(cache.bra_L_l2, cache.s, liouvillian_2)

        explicit_summation = 0
        for (u, state_u) in zip(1:4, TLS_Liouville_Space)
            for (v, state_v) in zip(1:4, TLS_Liouville_Space)
                loc = cache.bra_L_l2[v+4*(u-1)]
                if loc!=0
                    reduced_density_matrix = loc * optimizer.mpo.A[n,:,:,u] * B1 * optimizer.mpo.A[n,:,:,v] * B2
                    explicit_summation += tr(reduced_density_matrix)
                end
            end
        end
        
        temp_local_L_int += explicit_summation

        #println(explicit_summation/ρ_sample)
    end

    temp_local_L /= ρ_sample
    temp_local_L_int /= ρ_sample

    #display(temp_local_L_int)
    #error()

    return temp_local_L, temp_local_L_int
end



# OLD VERSION:
"""
function tensor_sweep_Lindblad!(sample::Projector, ρ_sample::T, optimizer::TDVP_H{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    cache = optimizer.workspace
    liouvillian = optimizer.l1
    liouvillian_2 = optimizer.l2

    # ONE-BODY TERMS:
    reduced_density_matrix = copy(optimizer.mpo)
    @tensor reduced_density_matrix.A[n,a,b,c] := liouvillian[c,d]*reduced_density_matrix.A[n,a,b,d]
    temp_local_L::T = 0
    for j in 1:params.N
        n = mod1(j, params.uc_size)
        mul!(cache.loc_1, cache.L_set[j], @view(reduced_density_matrix.A[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])]]))
        mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
        temp_local_L += tr(cache.loc_2)
    end

    # INTRA-LAYER TERMS:
    reduced_density_matrices = zeros(T, params.uc_size, params.χ, params.χ, 4, 4)
    reduced_density_matrix = zeros(T, params.χ, params.χ, 4, 4)
    for n in 1:params.uc_size
        A1 = optimizer.mpo.A[n,:,:,:]
        A2 = optimizer.mpo.A[mod1(n+1,params.uc_size),:,:,:]
        @tensor reduced_density_matrix[a,b,d,f] = liouvillian_2[c,e,d,f]*A1[a,u,c]*A2[u,b,e]
        reduced_density_matrices[n,:,:,:,:] = reduced_density_matrix
    end
    for j in 1:params.N-1
        if mod(j,params.uc_size)!=0
            n = mod1(j, params.uc_size)
            mul!(cache.loc_1, cache.L_set[j], @view(reduced_density_matrices[n,:,:,dINDEX[(sample.ket[j],sample.bra[j])],dINDEX[(sample.ket[j+1],sample.bra[j+1])]]))
            mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N-j)])
            temp_local_L += tr(cache.loc_2)
            #println(j,"-",j+1)
        end
    end

    # BOUNDARY-LAYER TERMS:
    for i in 1:params.uc_size
        j = 1+(i-1)*params.uc_size
        k = i*params.uc_size
        range = circshift(Array(1:params.N),-k)[1:end-params.uc_size]
        n = mod1(j, params.uc_size)
        m = mod1(k, params.uc_size)
        #k = mod1(j+params.uc_size, params.N)

        #println(j,"-",k)

        idx_j = dINDEX[(sample.ket[j],sample.bra[j])]
        idx_k = dINDEX[(sample.ket[k],sample.bra[k])]

        A1 = optimizer.mpo.A[n,:,:,idx_j]
        #A2 = optimizer.mpo.A[m,:,:,idx_k]

        #Compute matrix products inbetween sites j and k:
#        for i in range[1:params.uc_size-1]#j+1:k-1 # what if k<j?
        for l in j+1:k#-1#j+1:k-1 # what if k<j?
            p = mod1(l, params.uc_size)
            A1 = A1*optimizer.mpo.A[p,:,:,dINDEX[(sample.ket[l],sample.bra[l])]]
        end

        #Compute reduced denisty matrix:
#        reduced_density_matrix = A1*A2*liouvillian_2[idx_j, idx_k, idx_j, idx_k]
        reduced_density_matrix = A1*liouvillian_2[idx_j, idx_k, idx_j, idx_k]

        #Compute remaining matrix products:
#        for l in range#[1:end]
#            p = mod1(l, params.uc_size)
#            reduced_density_matrix = reduced_density_matrix*optimizer.mpo.A[p,:,:,dINDEX[(sample.ket[l],sample.bra[l])]]
#        end
        reduced_density_matrix = reduced_density_matrix * cache.R_set[(params.N+1-k)] * cache.L_set[j]

        temp_local_L += tr(reduced_density_matrix)
    end

    # CROSS-LAYER TERMS:
    for j in 1:params.N
        range = circshift(Array(1:params.N),-j)
        n = mod1(j, params.uc_size)
        k = mod1(j+params.uc_size, params.N)

        #println(j,"-",k)

        idx_j = dINDEX[(sample.ket[j],sample.bra[j])]
        idx_k = dINDEX[(sample.ket[k],sample.bra[k])]

        A1 = optimizer.mpo.A[n,:,:,idx_j]
        #A2 = optimizer.mpo.A[n,:,:,idx_k]

        #Compute matrix products inbetween sites j and k:
        for i in range[1:params.uc_size]#-1]#j+1:k-1 # what if k<j?
            m = mod1(i, params.uc_size)
            A1 = A1*optimizer.mpo.A[m,:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        end

        #Compute reduced denisty matrix:
        #reduced_density_matrix = A1*A2*liouvillian_2[idx_j, idx_k, idx_j, idx_k]        
        reduced_density_matrix = A1*liouvillian_2[idx_j, idx_k, idx_j, idx_k]

        #Compute remaining matrix products:
#        for i in range[params.uc_size+1:end-1]
#            m = mod1(i, params.uc_size)
#            reduced_density_matrix = reduced_density_matrix*optimizer.mpo.A[m,:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
#        end

        reduced_density_matrix = reduced_density_matrix * cache.R_set[(params.N+1-k)] * cache.L_set[j]

        temp_local_L += tr(reduced_density_matrix)
    end

    temp_local_L /= ρ_sample

    #error()

    return temp_local_L
end
"""


function tensor_update!(optimizer::TDVP{T}, sample::Projector) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.
    params=optimizer.params
    mpo=optimizer.mpo
    data=optimizer.optimizer_cache
    cache = optimizer.workspace

    local_L::T = 0
    local_L_int::T = 0
    l_int::T = 0

    #ρ_sample::T = trMPO(params, sample, mpo) 
    ρ_sample::T = tr(cache.R_set[params.N+1])
    cache.L_set = L_MPO_products!(cache.L_set, sample, mpo, params, cache)
    cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache, optimizer.mpo)./ρ_sample

    #display(sample)

    #Sweep lattice:
    local_L, local_L_int = tensor_sweep_Lindblad!(sample, ρ_sample, optimizer)

    #Add in Ising interaction terms:
    l_int = Ising_interaction_energy(optimizer.ising_op, sample, optimizer)

    #local_L += local_L_int
    local_L += l_int

    #display(sample)
    #display(local_L_int)
    #display(l_int)
    #sleep(5)
    #error()
    #local_L += l_int

    #display(sample)
    #display(local_L)
    #display(conj(cache.Δ))
    #sleep(10)
    #error()

    #Update joint ensemble average:
    data.L∂L.+=local_L*conj(cache.Δ)

    #Update disjoint ensemble average:
    data.ΔLL.+=conj(cache.Δ)

    #Mean local Lindbladian:
    data.mlL += local_L
    data.mlL2 += abs2(local_L)
    #data.mlL += local_L_int
    #data.mlL2 += l_int

    #return l_int
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
        #interaction_energy += tensor_update!(optimizer, sample) 
        tensor_update!(optimizer, sample) 

        #Update metric tensor:
        update_SR!(optimizer)
    end
    #display(optimizer.optimizer_cache.mlL)
    #display(optimizer.optimizer_cache.mlL2)
    #error()
    #sleep(4)
end