include("VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD
#using ProfileView


mpi_cache = set_mpi()

#Vincentini parameters: γ=1.0, J=0.5, h to be varied.

#Define constants:
const Jx= 0.0 #interaction strength
const Jy= 0.0 #interaction strength
const J = 0.5 #interaction strength
#const hx= 1.0 #longitudinal field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
const γ_d = 0.0 #spin decay rate
const α=0


#set values from command line optional parameters:
N = parse(Int64,ARGS[1])
hx = parse(Float64,ARGS[2])
χ = parse(Int64,ARGS[3])


params = Parameters(N,χ,Jx,Jy,J,hx,hz,γ,γ_d,α)


#Replace by an array of l1's!
const l1 = make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm)
const list_l1 = [l1 for _ in 1:N]
#display(list_l1)


N_MC::Int64 = parse(Int64,ARGS[4])
δ::Float64 = parse(Float64,ARGS[5])
N_iterations::Int64 = parse(Int64,ARGS[6])
ϵ::Float64 = parse(Float64,ARGS[7])

last_iteration_step::Int64 = 1
#ising_int = "SquareIsing"
ising_int = "TriangularIsing"

#Save parameters to file:
if mpi_cache.rank == 0
    
    start = now()
    
    #dir = "2D_Ising_decay_chi$(χ)_N$(N)_hx$(hx)"
    dir = "results/Triangular_Ising_decay_chi$(χ)_N$(N)_hx$(hx)"

    if isdir(dir)==false
        mkdir(dir)
        cd(dir)

        path = open("pwd.txt", "w")
        current_directory = dirname(pwd())
        println(path, current_directory)
        close(path)

        Random.seed!(0)
        A_init = zeros(ComplexF64, N,χ,χ,4)
        for i in 1:N
            A_init[i,:,:,1].=1.0
            A_init[i,:,:,2].=-1.0
            A_init[i,:,:,3].=-1.0
            A_init[i,:,:,4].=1.0
        end
        A = deepcopy(A_init)
        mpo = MPO(A)

        sampler = MetropolisSampler(N_MC, 5)
        optimizer = TDVP(sampler, mpo, list_l1, ϵ, params, ising_int)
        normalize_MPO!(params, optimizer)

        list_of_parameters = open("params.data", "w")
        #redirect_stdout(list_of_parameters)
        #display(list_of_parameters, params)
        #display(list_of_parameters, mpi_cache)
        #display(list_of_parameters, sampler)
        #display(list_of_parameters, optimizer)
        println(list_of_parameters, "N_MC\t\t", N_MC)
        println(list_of_parameters, "δ\t", δ)
        println(list_of_parameters, "\nN_iter\t", N_iterations)
        println(list_of_parameters, "Ising interaction: ", ising_int)
        close(list_of_parameters)
    else
        error()
        cd(dir)
        list_of_C = open("list_of_C.data", "r")
        last_iteration_step=countlines(list_of_C)+1

        ### NEED TO ALSO CHECK IF OTHER PARAMETERS ARE THE SAME BY EXPLICITLY COMPARING THE list_of_parameters FILES
    
        A_init = load("MPO_density_matrix.jld")["MPO_density_matrix"]
        A = reshape(A_init,χ,χ,4)
        
        A = normalize_MPO!(params, A)
    end
    L=0
    acc=0
    t0::Float64=0
    a::Float64=0
    b::Float64=0
    c::Float64=0
    d::Float64=0
else
    Random.seed!(mpi_cache.rank)
    A_init = zeros(ComplexF64, N,χ,χ,4)
    for i in 1:N
        A_init[i,:,:,1].=1.0
        A_init[i,:,:,2].=-1.0
        A_init[i,:,:,3].=-1.0
        A_init[i,:,:,4].=1.0
    end
    A = deepcopy(A_init)
    mpo = MPO(A)
end
MPI.bcast(last_iteration_step, mpi_cache.comm)


sampler = MetropolisSampler(N_MC, 5)
optimizer = TDVP(sampler, mpo, list_l1, ϵ, params, ising_int)
normalize_MPO!(params, optimizer)


if mpi_cache.rank == 0
    global t0 = time()

    mx = real(average_magnetization(params,mpo,sx))
    my = real(average_magnetization(params,mpo,sy))
    mz = real(average_magnetization(params,mpo,sz))

    list_of_C = open("list_of_C.data", "a")
    println(list_of_C, real(optimizer.optimizer_cache.mlL)/N)
    close(list_of_C)
    list_of_mag = open("list_of_mag.data", "a")
    println(list_of_mag, mx, ",", my, ",", mz)
    close(list_of_mag)
end

for k in last_iteration_step:N_iterations
    for i in 1:1
        if mpi_cache.rank == 0
            global a = time()
        end
        ComputeGradient!(optimizer)
        if mpi_cache.rank == 0
            global b = time()
        end
        MPI_mean!(optimizer,mpi_cache)
        if mpi_cache.rank == 0
            global c = time()
            Optimize!(optimizer,δ)
            global d = time()
        end
        MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)
        if mpi_cache.rank == 0
            global e = time()
        end
    end

    #Record observables:
    if mpi_cache.rank == 0

        list_of_C = open("list_of_C.data", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL)/N)
        close(list_of_C)

        mx = real(average_magnetization(params,mpo,sx))
        my = real(average_magnetization(params,mpo,sy))
        mz = real(average_magnetization(params,mpo,sz))
        list_of_mag = open("list_of_mag.data", "a")
        println(list_of_mag, mx, ",", my, ",", mz)
        close(list_of_mag)

        o = open("mem.out", "a")
        println(o, "k=$k: ", Base.Sys.free_memory())
        close(o)

        if mod(k,10)==1
            o = open("Ising_decay.out", "a")
            #redirect_stdout(o)
            println(o,"k=$k: ", real(optimizer.optimizer_cache.mlL)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))
            println(o,Base.Sys.free_memory())
            close(o)
        end

        save("MPO_density_matrix.jld", "MPO_density_matrix", optimizer.mpo.A)
        global f = time()
        list_of_times = open("list_of_times.data", "a")
        println(list_of_times, "Total for step ", k, ": ", round(f-a; sigdigits = 3))
        println(list_of_times, "In parts: ", round(b-a; sigdigits = 3), " ; ", round(c-b; sigdigits = 3), " ; ", round(d-c; sigdigits = 3), " ; ", round(e-d; sigdigits = 3), " ; ", round(f-e; sigdigits = 3))
        close(list_of_times)
    end

    if mod(k,10)==0
        GC.gc()
    end
end

if mpi_cache.rank == 0
    list_of_times = open("list_of_times.data", "a")
    println(list_of_times, "Sum total: ", f-t0)
    close(list_of_times)
end