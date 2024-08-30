include("../VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD
using SparseArrays


mpi_cache = set_mpi()

#Set parameters:
Jx= 0.0 #interaction strength
Jy= 0.25 #interaction strength
J = 0.0 #interaction strength
hx= 0.5 #transverse field strength
hz= 0.0 #longitudinal field strength
γ = 1.0 #spin decay rate
uc_size = 3
N = uc_size^2 #number of spins
α = 0
γ_d = 0

#Set hyperparameters:
χ = 10 #MPO bond dimension
N_MC = 1000 #number of Monte Carlo samples
burn_in = 2 #Monte Carlo burn-in
δ = 0.05 #step size
ϵ = 0.01
N_iterations = 1500
ising_int="SquareIsing"

params = Parameters(N,χ,Jx,Jy,J,hx,hz,γ,γ_d,α,uc_size)


#Define one-body Lindbladian operator:
l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*sm)#*adjoint(smx))

#l2 = Jx*make_two_body_Lindblad_Hamiltonian(sx, sx)
#l2 = reshape(l2, 4,4,4,4)

l2 = Jx*make_two_body_Lindblad_Hamiltonian(sx, sx)+Jy*make_two_body_Lindblad_Hamiltonian(sy, sy)+J*make_two_body_Lindblad_Hamiltonian(sz, sz)
display(sparse(l2))
l2 = reshape(l2, 4,4,4,4)
#l2 = permutedims(l2, (1,3,2,4))

#display(l2)
#display(size(l2))
#l2r = reshape(l2, 4,4,4,4)
#display(l2r)
#display(size(l2r))error()


#Save parameters to file:
dir = "Ising_decay_chi$(χ)_N$(N)_J$(J)_hx$(hx)_hz$(hz)_γ$(γ)"

if mpi_cache.rank == 0
    if isdir(dir)==true
        error("Directory already exists")
    end
    mkdir(dir)
    cd(dir)
end

Random.seed!(4)#(mpi_cache.rank)
A_init = zeros(ComplexF64, uc_size,χ,χ,4)
for n in 1:uc_size
    A_init[n,:,:,1].=1.0
    A_init[n,:,:,2].=1.0
    A_init[n,:,:,3].=1.0
    A_init[n,:,:,4].=1.0
#    A_init[n,:,:,1].=1.0
#    A_init[n,:,:,2].=1.0
#    A_init[n,:,:,3].=1.0
#    A_init[n,:,:,4].=rand(ComplexF64)
end
mpo = MPO(A_init)

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in)
optimizer = TDVP_H(sampler, mpo, l1, l2, ϵ, params)
#optimizer = TDVP(sampler, mpo, l1, ϵ, params, ising_int)
normalize_MPO!(params, optimizer)

"""
if mpi_cache.rank == 0
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    #display(optimizer)
    println("\nN_iter\t", N_iterations)
    println("δ\t\t", δ)
    close(list_of_parameters)
end
"""

for k in 1:N_iterations

    #Optimize MPO:
    TensorComputeGradient!(optimizer)
    MPI_mean!(optimizer, mpi_cache)
    if mpi_cache.rank == 0
        Optimize!(optimizer, δ)
    end
    MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)

    if mpi_cache.rank == 0
        #Calculate steady-state magnetizations:
#        mx = real.( tensor_magnetization(2,params,mpo,sx) )
#        my = real.( tensor_magnetization(2,params,mpo,sy) )
#        mz = real.( tensor_magnetization(2,params,mpo,sz) )

        Z = tensor_purity(params,mpo)

        mx=0.0
        my=0.0
        mz=0.0
        for n in 1:uc_size
            mx += real.( tensor_magnetization(n,params,mpo,sx) )
            my += real.( tensor_magnetization(n,params,mpo,sy) )
            mz += real.( tensor_magnetization(n,params,mpo,sz) )
        end
        mx/=uc_size
        my/=uc_size
        mz/=uc_size

        list_of_C = open("C.out", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
        close(list_of_C)

        list_of_obs = open("obs.out", "a")
        println(list_of_obs, mx, ",", my, ",", mz, ",", Z)
        close(list_of_obs)
    end

    #if mod(k,10)==0
    #    GC.gc()
    #end
end
