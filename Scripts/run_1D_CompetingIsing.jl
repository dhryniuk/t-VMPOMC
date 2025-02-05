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
Jx= 0.0 #interaction strength
Jy= 0.0 #interaction strength
Jz= 0.0 #interaction strength
J1= 0.5 #interaction strength
J2= -1.0 #interaction strength
hx= 0.5 #transverse field strength
hz= 0.0 #longitudinal field strength
γ = 1.0 #spin decay rate
N = parse(Int64,ARGS[1]) #number of spins
α1 = 3.0
α2 = 6.0
γ_d = 0

#Set hyperparameters:
χ = parse(Int64,ARGS[2]) #MPO bond dimension
uc_size = parse(Int64,ARGS[3])
N_MC = parse(Int64,ARGS[4]) #number of Monte Carlo samples
burn_in = 2 #Monte Carlo burn-in
sweeps = 2
T = parse(Float64,ARGS[5])
ϵ_shift = parse(Float64,ARGS[6])
ϵ_SNR = parse(Float64,ARGS[7])
ϵ_tol = parse(Float64,ARGS[8])
max_τ = parse(Float64,ARGS[9])
ising_int = "CompetingIsing"
τ = 10^(-8)

params = Parameters(N,χ,Jx,Jy,Jz,J1,J2,hx,hz,γ,γ_d,α1,α2,uc_size)

#Define one-body Lindbladian operator:
l1 = make_one_body_Lindbladian(hx*sx+hz*sz, sqrt(γ)*sm)

mpo = MPO("-y", params, mpi_cache)

#Define sampler and optimizer:
sampler = MetropolisSampler(N_MC, burn_in, sweeps, params)
optimizer = TDVP(sampler, mpo, l1, τ, ϵ_shift, ϵ_SNR, ϵ_tol, params, ising_int)
NormalizeMPO!(params, optimizer)


last_iteration_step = 1

#Save parameters to file:
dir = "results/LRIsing_1D_uc$(uc_size)_chi$(χ)_N$(N)_J1$(J1)_α1$(α1)_J2$(J2)_α2$(α2)_hx$(hx)_hz$(hz)_γ$(γ)"
if mpi_cache.rank == 0
    if isdir(dir)==true
        cd(dir)
        optimizer = load("optimizer.jld", "optimizer")
        list_of_C = open("C.out", "r")
        last_iteration_step=countlines(list_of_C)+1
    else    
        mkdir(dir)
        cd(dir)
    end
end


if mpi_cache.rank == 0 && last_iteration_step == 1
    
    #Save parameters to parameter file:
    list_of_parameters = open("Ising_decay.params", "w")
    redirect_stdout(list_of_parameters)
    display(params)
    display(sampler)
    display(optimizer)
    close(list_of_parameters)

    mx = 0.0
    my = 0.0
    mz = 0.0
    for n in 1:uc_size
        global mx += real.( tensor_magnetization(n,params,mpo,sx) )
        global my += real.( tensor_magnetization(n,params,mpo,sy) )
        global mz += real.( tensor_magnetization(n,params,mpo,sz) )
    end
    mx/=uc_size
    my/=uc_size
    mz/=uc_size

    list_of_C = open("C.out", "a")
    println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
    close(list_of_C)

    list_of_obs = open("obs.out", "a")
    println(list_of_obs, mx, ",", my, ",", mz)
    close(list_of_obs)

    Cxx = tensor_correlation(1,2,sx,sx,params,mpo) - mx^2
    Cyy = tensor_correlation(1,2,sy,sy,params,mpo) - my^2
    Czz = tensor_correlation(1,2,sz,sz,params,mpo) - mz^2

    list_of_obs = open("corr2.out", "a")
    println(list_of_obs, Cxx, ",", Cyy, ",", Czz)
    close(list_of_obs)

    Cxx = tensor_correlation(1,3,sx,sx,params,mpo) - mx^2
    Cyy = tensor_correlation(1,3,sy,sy,params,mpo) - my^2
    Czz = tensor_correlation(1,3,sz,sz,params,mpo) - mz^2

    list_of_obs = open("corr3.out", "a")
    println(list_of_obs, Cxx, ",", Cyy, ",", Czz)
    close(list_of_obs)

    M_sq = ( modulated_magnetization_TI(0.0, params, mpo, sz) )#^0.5
    M_stag = ( modulated_magnetization_TI(π, params, mpo, sz))#^0.5
    M_mod = ( modulated_magnetization_TI(2π/params.N, params, mpo, sz))#^0.5

    list_of_obs = open("ssf.out", "a")
    println(list_of_obs, M_sq, ",", M_stag, ",", M_mod)
    close(list_of_obs)

    list_of_times = open("times.out", "a")
    println(list_of_times, 0.0)
    close(list_of_times)
end

MPI.Barrier(mpi_cache.comm)
if mpi_cache.rank != 0
    cd(dir)
end
MPI.bcast(last_iteration_step, 0, mpi_cache.comm)
MPI.bcast(optimizer.τ, 0, mpi_cache.comm)
MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)

current_time::Float64 = parse(Float64, readlines("times.out")[end])
while current_time<T

    global last_iteration_step+=1

    #EulerStep!(optimizer, mpi_cache)
    AdaptiveHeunStepCapped!(max_τ, optimizer, mpi_cache)

    if mpi_cache.rank == 0

        mpo = optimizer.mpo
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
    
        save("optimizer.jld", "optimizer", optimizer)

        list_of_C = open("C.out", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL2)/N)
        close(list_of_C)
    
        list_of_obs = open("obs.out", "a")
        println(list_of_obs, mx, ",", my, ",", mz)
        close(list_of_obs)
    
        Cxx = tensor_correlation(1,2,sx,sx,params,mpo) - mx^2
        Cyy = tensor_correlation(1,2,sy,sy,params,mpo) - my^2
        Czz = tensor_correlation(1,2,sz,sz,params,mpo) - mz^2
    
        list_of_obs = open("corr2.out", "a")
        println(list_of_obs, Cxx, ",", Cyy, ",", Czz)
        close(list_of_obs)
    
        Cxx = tensor_correlation(1,3,sx,sx,params,mpo) - mx^2
        Cyy = tensor_correlation(1,3,sy,sy,params,mpo) - my^2
        Czz = tensor_correlation(1,3,sz,sz,params,mpo) - mz^2
    
        list_of_obs = open("corr3.out", "a")
        println(list_of_obs, Cxx, ",", Cyy, ",", Czz)
        close(list_of_obs)

        M_sq = ( modulated_magnetization_TI(0.0, 0.0, params, mpo, sz) )#^0.5
        M_stag = ( modulated_magnetization_TI(π, π, params, mpo, sz))#^0.5
        M_mod = ( modulated_magnetization_TI(2π/params.uc_size, 2π/params.uc_size, params, mpo, sz))#^0.5

        list_of_obs = open("ssf.out", "a")
        println(list_of_obs, M_sq, ",", M_stag, ",", M_mod)
        close(list_of_obs)

        global current_time += optimizer.τ
        list_of_times = open("times.out", "a")
        println(list_of_times, current_time)
        close(list_of_times)

        save("optimizer_backup.jld", "optimizer", optimizer)
    end

    GC.gc()
end

MPI.Barrier(mpi_cache.comm)

error("Exiting")
