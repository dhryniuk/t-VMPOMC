function TensorSweepLindblad!(sample::Projector, ρ_sample::T, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    ws = optimizer.workspace
    liouvillian = optimizer.l1

    reduced_density_matrix::MPO{T} = deepcopy(optimizer.mpo)
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

    reduced_density_matrix = deepcopy(optimizer.mpo)
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

    # Comment out for OBC:
    #"""
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
    #"""

    temp_local_L /= ρ_sample

    return temp_local_L, 0
end

function TensorSweepLindblad!(sample::Projector, ρ_sample::T, optimizer::TDVP_H{T}) where {T<:Complex{<:AbstractFloat}} 
    params = optimizer.params
    ws = optimizer.workspace
    liouvillian = optimizer.l1
    liouvillian_2 = reshape(optimizer.l2, 4*4,4*4)

    # ONE-BODY TERMS:
    reduced_density_matrix = deepcopy(optimizer.mpo)
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