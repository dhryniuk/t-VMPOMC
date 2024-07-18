### NEED TO REFACTOR:


#Sweeps lattice from right to left
function Metropolis_sweep_left(sample::Projector, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 

    A = optimizer.mpo.A
    params = optimizer.params
    cache=optimizer.workspace

    acc=0
    cache.R_set[1] = Matrix{T}(I, params.χ, params.χ)
    cache.C_mat = cache.L_set[params.N+1]
    C = tr(cache.C_mat) #current probability amplitude

    for i::UInt16 in params.N:-1:1
        sample_p = Projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i],sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(cache.Metro_1,cache.L_set[i],@view(A[i,:,:,draw]))
        mul!(cache.Metro_2,cache.Metro_1,cache.R_set[params.N+1-i])
        P=tr(cache.Metro_2) #proposal probability amplitude
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = Projector(sample_p)
            acc+=1
        end
        mul!(cache.R_set[params.N+2-i], @view(A[i,:,:,1+2*sample.ket[i]+sample.bra[i]]), cache.R_set[params.N+1-i])
        mul!(cache.C_mat, cache.L_set[i], cache.R_set[params.N+2-i])
        C = tr(cache.C_mat) #update current probability amplitude
    end
    return sample, acc
end

#Sweeps lattice from left to right
function Metropolis_sweep_right(sample::Projector, optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 

    A = optimizer.mpo.A
    params = optimizer.params
    cache=optimizer.workspace

    acc=0
    cache.L_set[1] = Matrix{T}(I, params.χ, params.χ)
    cache.C_mat = cache.R_set[params.N+1]
    C = tr(cache.C_mat) #current probability amplitude

    for i::UInt16 in 1:params.N
        sample_p = Projector(sample)
        draw = draw_excluded(dINDEX[(sample.ket[i],sample.bra[i])])
        (sample_p.ket[i], sample_p.bra[i]) = dREVINDEX[draw]
        mul!(cache.Metro_1,cache.L_set[i],@view(A[i,:,:,draw]))
        mul!(cache.Metro_2,cache.Metro_1,cache.R_set[params.N+1-i])
        P=tr(cache.Metro_2) #proposal probability amplitude
        metropolis_prob = real((P*conj(P))/(C*conj(C)))
        if rand() <= metropolis_prob
            sample = Projector(sample_p)
            acc+=1
        end
        mul!(cache.L_set[i+1], cache.L_set[i], @view(A[i,:,:,1+2*sample.ket[i]+sample.bra[i]]))
        mul!(cache.C_mat, cache.L_set[i+1], cache.R_set[params.N+1-i])
        C = tr(cache.C_mat) #update current probability amplitude
    end
    return sample, acc
end

function MPO_Metropolis_burn_in(optimizer::TDVPl1{T}) where {T<:Complex{<:AbstractFloat}} 

    A=optimizer.mpo.A
    mpo=optimizer.mpo
    params=optimizer.params
    cache=optimizer.workspace
    
    # Initialize random sample and calculate L_set for that sample:
    sample::Projector = Projector(rand(Bool, params.N),rand(Bool, params.N))
    #sample::Projector = Projector(Bool[0], Bool[0])
    cache.L_set = L_MPO_products!(cache.L_set, sample, mpo, params, cache)

    #display(sample)
    # Perform burn_in:
    for _ in 1:optimizer.sampler.burn
        sample,_ = Metropolis_sweep_left(sample,optimizer)
        #display(sample)
        sample,_ = Metropolis_sweep_right(sample,optimizer)
        #display(sample)
    end
    return sample
end
