function Reconfigure!(optimizer::TDVP{T}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC

    #Compute metric tensor:
    data.S ./= N_MC

    #Regularize the metric tensor:
    data.S += optimizer.ϵ_SNR*Matrix{Int}(I, size(data.S))

    #Reconfigure gradient:
    grad = data.L∂L/N_MC
    flat_grad = reshape(grad,prod(size(grad)))
    flat_grad = inv(data.S)*flat_grad

    data.∇ = reshape(flat_grad,size(data.∇))
end

function f_variance(V, local_estimators, arr_gradients, params)

    # Following Schmitt and Heyl:

    gradients = [reshape(arr_gradients[i,:], 4*params.χ^2*params.uc_size) for i in 1:size(arr_gradients, 1)]
    mean_gradient = mean(gradients)
    mean_local_estimator = mean(local_estimators)

    Q = [V' * (gradients[i]-mean_gradient) for i=1:length(gradients)]
    E_loc = local_estimators .- mean_local_estimator

    f_var = similar(Q[1])
    diff = similar(f_var)
    f_var .= 0
    diff .= 0

    for (i,_) in enumerate(local_estimators)
        g = ((Q[i])' * E_loc[i])'
        diff += g
        f_var += abs2.(g)
    end
    f_var /= length(local_estimators)
    diff /= length(local_estimators)
    f_var -= abs2.(diff) 

    return f_var, abs2.(diff)
end

function Reconfigure!(optimizer::TDVP{ComplexF64}, local_estimators::Vector{ComplexF64}, gradients::Array{ComplexF64,2}) where {T<:Complex{<:AbstractFloat}} #... the gradient tensor
    data = optimizer.optimizer_cache
    N_MC = optimizer.sampler.N_MC
    params = optimizer.params

    # Compute metric tensor:
    data.S./=N_MC

    # Obtain gradient vector:
    grad = (data.L∂L)/N_MC
    #grad = (data.L∂L - data.ΔLL*data.mlL)/N_MC
    flat_grad = reshape(grad, prod(size(grad)))

    σ, _ = eigen(data.S)
    λ = 2*abs(minimum(real.(σ)))

    #flat_grad = inv(data.S + (λ+optimizer.ϵ_shift)*Matrix{Int}(I, size(data.S)))*flat_grad

    # ϵ-shift regulator:
    σ, V = eigen(data.S + (λ+optimizer.ϵ_shift)*Matrix{Int}(I, size(data.S)))
    flat_grad = V'*flat_grad
    #flat_grad = V'*flat_grad*V

    # Moore-Penrose pseudoinverse regulator:
    if optimizer.ϵ_SNR!=0.0
        f_var, diff = f_variance(V, local_estimators, gradients, params)
        SNR = sqrt.(diff)./sqrt.(f_var./N_MC)
        σ_inv = ( (σ).*reshape((1 .+ (optimizer.ϵ_SNR./SNR).^1), 4*params.χ^2*params.uc_size) ).^(-1.0)
        flat_grad = V*diagm(σ_inv)*flat_grad
    end

    data.∇ = reshape(flat_grad, size(data.∇))

    ### Doesn't seem to matter, but re-zero anyway:
    data.S = zeros(ComplexF64, 4*params.χ^2*params.uc_size, 4*params.χ^2*params.uc_size)
end
