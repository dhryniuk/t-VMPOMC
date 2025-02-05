#Sweeps lattice from right to left
function MetropolisSweepLeft!(sample::Projector, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params
    ws = optimizer.workspace
    acc = 0

    ws.R_set[1] = Matrix{T}(I, params.χ, params.χ)  # Identity matrix
    ws.C_mat = ws.L_set[params.N+1]
    C = tr(ws.C_mat)  # current probability amplitude
    
    sample_ket = sample.ket
    sample_bra = sample.bra
    
    @inbounds for i in params.N:-1:1
        n = mod1(i, params.uc_size)

        current_index = dINDEX[(sample_ket[i], sample_bra[i])]
        draw = draw_excluded(current_index)
        new_ket, new_bra = dREVINDEX[draw]
        
        mul!(ws.Metro_1, ws.L_set[i], view(A, n, :, :, draw))
        mul!(ws.Metro_2, ws.Metro_1, ws.R_set[params.N+1-i])
        P = tr(ws.Metro_2)  # proposal probability amplitude
        
        metropolis_prob = abs2(P/C)
        if rand() <= metropolis_prob
            sample_ket[i] = new_ket
            sample_bra[i] = new_bra
            acc += 1
            C = P
        else
            draw = current_index
        end
        
        mul!(ws.R_set[params.N+2-i], view(A, n, :, :, draw), ws.R_set[params.N+1-i])
        mul!(ws.C_mat, ws.L_set[i], ws.R_set[params.N+2-i])
        C = tr(ws.C_mat)  # update current probability amplitude
    end
    
    sample.ket = sample_ket
    sample.bra = sample_bra
    return sample, acc
end

#Sweeps lattice from left to right
function MetropolisSweepRight!(sample::Projector, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    A = optimizer.mpo.A
    params = optimizer.params
    ws = optimizer.workspace
    acc = 0
    
    ws.L_set[1] = Matrix{T}(I, params.χ, params.χ)
    ws.C_mat = ws.R_set[params.N+1]
    C = tr(ws.C_mat)  # current probability amplitude
    
    sample_ket = sample.ket
    sample_bra = sample.bra
    
    @inbounds for i in 1:params.N
        n = mod1(i, params.uc_size)

        current_index = dINDEX[(sample_ket[i], sample_bra[i])]
        draw = draw_excluded(current_index)
        new_ket, new_bra = dREVINDEX[draw]
        
        mul!(ws.Metro_1, ws.L_set[i], view(A, n, :, :, draw))
        mul!(ws.Metro_2, ws.Metro_1, ws.R_set[params.N+1-i])
        P = tr(ws.Metro_2)  # proposal probability amplitude
        
        metropolis_prob = abs2(P/C)
        if rand() <= metropolis_prob
            sample_ket[i] = new_ket
            sample_bra[i] = new_bra
            acc += 1
            C = P
        else
            draw = current_index
        end
        
        mul!(ws.L_set[i+1], ws.L_set[i], view(A, n, :, :, draw))
        mul!(ws.C_mat, ws.L_set[i+1], ws.R_set[params.N+1-i])
        C = tr(ws.C_mat)  # update current probability amplitude
    end
    
    sample.ket = sample_ket
    sample.bra = sample_bra
    return sample, acc
end

function MPO_Metropolis_burn_in!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    A=optimizer.mpo.A
    params=optimizer.params
    ws=optimizer.workspace

    # Initialize random sample and calculate L_set for that sample:
    sample::Projector = Projector(rand(Bool, params.N),rand(Bool, params.N))
    ws.L_set = L_MPO_products!(ws.L_set, sample, optimizer.mpo, params, ws)

    # Perform burn_in:
    for _ in 1:optimizer.sampler.burn
        sample,_ = MetropolisSweepLeft!(sample,optimizer)
        sample,_ = MetropolisSweepRight!(sample,optimizer)
    end
    return sample
end

function MetropolisSweepLeft!(sample::Projector, sweeps::Int64, optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} 
    
    sample,_ = MetropolisSweepLeft!(sample,optimizer)

    # Perform burn_in:
    for i in 1:sweeps-1
        sample,_ = MetropolisSweepRight!(sample,optimizer)
        sample,_ = MetropolisSweepLeft!(sample,optimizer)
    end
    return sample, 0
end