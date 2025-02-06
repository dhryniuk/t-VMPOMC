include("../VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD


mpi_cache = set_mpi()

#Set parameters:
J1= 0.5 #interaction strength
J2= -1.0 #interaction strength
α1 = 3.0 #exponent for J1
α2 = 6.0 #exponent for J2
hx= 0.5 #transverse field strength
hz= 0.0 #longitudinal field strength
γ = 1.0 #spin decay rate
N = 10 #number of spins

#Set hyperparameters:
ising_int = "CompetingIsing" #Ising interaction type
χ = 6 #MPO bond dimension
uc_size = 1 #unit cell size
N_MC = 1000 #number of Monte Carlo samples
burn_in = 2 #Monte Carlo burn-in
sweeps = 2 #number of Monte Carlo sweeps
T = 2.0 #total simulation time
ϵ_shift = 10^(-9) #shift regularization
ϵ_SNR = 10^(-3) #SNR regularization
ϵ_tol = 0.05 #tolerance for truncation error
max_τ = 0.1 #maximum time step
τ = 10^(-8) #initial time step

params = Parameters(N,χ,0,0,0,J1,J2,hx,hz,γ,0,α1,α2,uc_size)


#Define one-body Lindbladian operator:
l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*sm)

#Define initial MPO:
mpo = MPO("-y", params, mpi_cache)

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in, sweeps, params)
optimizer = TDVP(sampler, mpo, l1, τ, ϵ_shift, ϵ_SNR, ϵ_tol, params, ising_int)
NormalizeMPO!(params, optimizer)

last_iteration_step = 1

#Save parameters to file:
dir = "Long_range_Ising_decay"
mkdir(dir)
cd(dir)


if mpi_cache.rank == 0 && last_iteration_step == 1
    
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    display(optimizer)
    close(list_of_parameters)

    mx, my, mz = measure_magnetizations(params, mpo)

    list_of_obs = open("magnetizations.out", "a")
    println(list_of_obs, mx, ",", my, ",", mz)
    close(list_of_obs)

    list_of_times = open("times.out", "a")
    println(list_of_times, 0.0)
    close(list_of_times)
end

# Synchronize all MPI processes before proceeding:
MPI.Barrier(mpi_cache.comm)

# For non-root processes, change the working directory:
if mpi_cache.rank != 0
    cd(dir)
end

# Broadcast shared parameters from the root process to all other processes:
MPI.bcast(last_iteration_step, 0, mpi_cache.comm)
MPI.bcast(optimizer.τ, 0, mpi_cache.comm)
MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)

# Initialize simulation time:
current_time = 0.0

# Main simulation loop: run until the simulation time reaches target T:
while current_time<T

    global last_iteration_step+=1

    # Update the optimizer state using an adaptive time-stepping method:
    AdaptiveHeunStepCapped!(max_τ, optimizer, mpi_cache)

    # Only the root process handles measurement output:
    if mpi_cache.rank == 0

        mx, my, mz = measure_magnetizations(params, optimizer.mpo)

        list_of_obs = open("magnetizations.out", "a")
        println(list_of_obs, mx, ",", my, ",", mz)
        close(list_of_obs)

        global current_time += optimizer.τ
        list_of_times = open("times.out", "a")
        println(list_of_times, current_time)
        close(list_of_times)
    end
end

MPI.Barrier(mpi_cache.comm)

error("Exiting")
