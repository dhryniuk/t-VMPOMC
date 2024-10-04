function Reconfigure!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC
    ϵ = optimizer.ϵ
    params = optimizer.params

    #Compute metric tensor:
    data.S./=N_MC
    data.avg_G./=N_MC
    conj_avg_G = conj(data.avg_G)
    data.S-=data.avg_G*transpose(conj_avg_G)

    #Regularize the metric tensor:
    data.S+=ϵ*Matrix{Int}(I, size(data.S))

    #Reconfigure gradient:
    grad = (data.L∂L-data.ΔLL)/N_MC
    flat_grad = reshape(grad,prod(size(grad)))
    flat_grad = inv(data.S)*flat_grad

    data.∇ = reshape(flat_grad,size(data.∇))
end

function f_variance(V, local_estimators, arr_gradients, params)
    #vec_gradients = [arr_gradients[i, :, :, :, :] for i in 1:size(arr_gradients, 1)]

    #display(arr_gradients)
    #sleep(5)
    #error()

    #vec_gradients = [arr_gradients[i, :, :, :, :] for i in 1:6]

    #display(arr_gradients)
    #sleep(5)
    #display(typeof(vec_gradients))
    #display(vec_gradients)
    #sleep(5)
    #error()

    gradients = [reshape(arr_gradients[i,:], 4*params.χ^2*params.uc_size) for i in 1:size(arr_gradients, 1)]
    #gradients = [reshape(vec_gradients[i], (1,4*params.χ^2*params.uc_size))*V for i in 1:length(vec_gradients)]

    #=
    display(gradients)
    sleep(5)
    display(typeof(gradients))
    sleep(5)
    display(size(gradients))
    sleep(5)
    display(gradients[1])
    sleep(5)
    display(size(gradients[1]))
    error()
    =#

    mean_gradient = zeros(ComplexF64, size(gradients[1]))
    for i in 1:length(gradients)
        mean_gradient += gradients[i]
    end
    mean_gradient./=length(gradients)
    #mean_gradient = mean(gradients)
    mean_local_estimator = mean(local_estimators)

    Q = [V' * (gradients[i]-mean_gradient) for i=1:length(gradients)]
    E_loc = local_estimators .- mean_local_estimator

    #display(arr_gradients)
    #display(gradients[1,:])
    #display(size(gradients[1,:]))
    #display(typeof(gradients[1,:]))
    #sleep(5)
    #error()

    #display(Q[1])
    #sleep(5)
    #error()

    f_var = similar(Q[1])
    diff = similar(f_var)
    f_var .= 0
    diff .= 0

    #display(f_var)
    #sleep(5)
    #error()

    for (i,_) in enumerate(local_estimators)
        #Q = conj.(gradients[i])
        g = ((Q[i])' * E_loc[i])'

        #display(ρ)
        #sleep(5)
        #error()
        diff += g
        f_var += abs2.(g)
        #f_var += conj.(ρ).*ρ
    end
    f_var/=length(local_estimators)
    diff/=length(local_estimators)

    #f_var -= conj.(diff).*diff    
    f_var -= abs2.(diff) #diff'*diff  

    #return f_var, conj.(diff).*diff
    return f_var, abs2.(diff)

    #=
    f_var = similar(gradients[1])
    diff = similar(f_var)
    f_var .= 0
    diff .= 0

    for (i,_) in enumerate(local_estimators)
        Q = conj.(gradients[i])
        f = (Q .- conj.(mean_gradient)) * (local_estimators[i] - mean_local_estimator)
        diff += f
        f_var += conj.(f).*f
    end
    f_var/=length(local_estimators)
    diff/=length(local_estimators)

    f_var -= conj.(diff).*diff    

    return f_var, conj.(diff).*diff
    =#
end

function Reconfigure!(optimizer::TDVP{ComplexF64}, local_estimators::Vector{ComplexF64}, gradients::Array{ComplexF64,2}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
#function Reconfigure!(optimizer::TDVP{ComplexF64}, local_estimators::Vector{ComplexF64}, gradients::Array{ComplexF64, 5}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC
    params = optimizer.params

    # Compute metric tensor:
    data.S./=N_MC
    data.avg_G./=N_MC
    conj_avg_G = conj(data.avg_G)
    data.S -= conj_avg_G*transpose(data.avg_G)

    # Obtain gradient vector:
    grad = (data.L∂L-data.ΔLL)/N_MC
    flat_grad = reshape(grad, prod(size(grad)))

    # ϵ-shift regulator:
    σ, V = eigen(data.S+optimizer.ϵ*Matrix{Int}(I, size(data.S)))
    flat_grad = V'*flat_grad

    # Moore-Penrose pseudoinverse regulator:
    if optimizer.ϵ_SNR!=0.0
        f_var, diff = f_variance(V, local_estimators, gradients, params)
        SNR = sqrt.(diff)./sqrt.(f_var./N_MC)
        σ_inv = ( (σ).*reshape((1 .+ (optimizer.ϵ_SNR./SNR).^1), 4*params.χ^2*params.uc_size) ).^(-1.0)
        flat_grad = V*diagm(σ_inv)*flat_grad
    end
    #cutoff = 10^(-8)
    #for i in 1:length(σ)
    #    if abs.(σ[i])<cutoff
    #        σ_inv[i] = 0.0
    #    end
    #end 

    data.∇ = reshape(flat_grad, size(data.∇))
end