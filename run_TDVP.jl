include("VMPOMC.jl")
using .VMPOMC
using NPZ
using Plots
using LinearAlgebra
import Random
using MPI
using Dates
using JLD


mpi_cache = set_mpi()

#Vincentini parameters: γ=1.0, J=0.5, h to be varied.

#Define constants:
const Jx= 0.0 #interaction strength
const Jy= 0.0 #interaction strength
const J = 0.0 #interaction strength
#const hx= 1.0 #transverse field strength
const hz= 0.0 #transverse field strength
const γ = 1.0 #spin decay rate
#const γ_d = 0.0 #spin decay rate
const α=0#0000
#const N=10
#χ=12 #bond dimension
#const burn_in = 0


#set values from command line optional parameters:
N = parse(Int64,ARGS[1])
const γ_d = 0
hx = parse(Float64,ARGS[2])
χ = parse(Int64,ARGS[3])


params = Parameters(N,χ,Jx,Jy,J,hx,hz,γ,γ_d,α)


#Replace by an array of l1's!
const l1 = make_one_body_Lindbladian(hx*sx+hz*sz,sqrt(γ)*sm)
const list_l1 = [l1 for _ in 1:N]

display(l1)

N_MC::Int64 = 1000#10*4*χ^2
δ::Float64 = 0.01
F::Float64 = 1.0#0.9999
ϵ::Float64 = parse(Float64,ARGS[4])
N_iterations::Int64 = 1000
last_iteration_step::Int64 = 1

#Save parameters to file:
if mpi_cache.rank == 0
    
    start = now()
    
    dir = "Ising_decay_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d)"

    if isdir(dir)==false
        mkdir(dir)
        cd(dir)

        path = open("pwd.txt", "w")
        current_directory = dirname(pwd())
        println(path, current_directory)
        close(path)

        Random.seed!(0)
        #A_init = [zeros(ComplexF64, χ,χ,4) for _ in 1:N]
        A_init = zeros(ComplexF64, N,χ,χ,4)
        v1 = [0.98,0.02,0.02,0.02]
        for i in 1:χ
            A_init[1,1,i,:]=v1
            A_init[N,i,1,:]=v1
            for n in 2:N-1
                for j in 1:χ
                    A_init[n,j,i,:]=v1
                end
            end
        end
        A = deepcopy(A_init)
        mpo = MPO(A)

        sampler = MetropolisSampler(N_MC, 5)
        optimizer = TDVP(sampler, mpo, list_l1, ϵ, params)
        normalize_MPO!(params, optimizer)
    else
        error()
        cd(dir)
        list_of_C = open("list_of_C_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "r")
        last_iteration_step=countlines(list_of_C)+1

        ### NEED TO ALSO CHECK IF OTHER PARAMETERS ARE THE SAME BY EXPLICITLY COMPARING THE list_of_parameters FILES
    
        A_init = load("MPO_density_matrix_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).jld")["MPO_density_matrix"]
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
    A = [zeros(ComplexF64, χ,χ,4) for _ in 1:N]
    mpo = MPO(A)
end
MPI.bcast(last_iteration_step, mpi_cache.comm)
#println(typeof(mpo.A))
#println(eltype(mpo.A))
#MPI.Bcast!(mpo.A, 0, mpi_cache.comm)

sampler = MetropolisSampler(N_MC, 5)
optimizer = TDVP(sampler, mpo, list_l1, ϵ, params)


if mpi_cache.rank == 0
    global t0 = time()

    #Af = reshape(optimizer.mpo.A,χ,χ,2,2) 
    #Af_dagger = conj.(permutedims(Af,[1,2,4,3]))

    mx = real(tensor_calculate_magnetization(params,mpo,sx))
    my = real(tensor_calculate_magnetization(params,mpo,sy))
    mz = real(tensor_calculate_magnetization(params,mpo,sz))

    list_of_C = open("list_of_C_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "a")
    println(list_of_C, real(optimizer.optimizer_cache.mlL)/N)
    close(list_of_C)
    list_of_mag = open("list_of_mag_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "a")
    println(list_of_mag, mx, ",", my, ",", mz)
    close(list_of_mag)
end
for k in last_iteration_step:N_iterations
    for i in 1:1
        if mpi_cache.rank == 0
            global a = time()
        end
        ComputeGradient!(optimizer)
        #ComputeGradient!(optimizer, basis)
        if mpi_cache.rank == 0
            global b = time()
        end
        MPI_mean!(optimizer,mpi_cache)
        if mpi_cache.rank == 0
            global c = time()
            Optimize!(optimizer,δ*F^(k))
            #Optimize!(optimizer,basis,δ*F^(k))
            global d = time()
        end
        MPI.Bcast!(optimizer.mpo.A, 0, mpi_cache.comm)
        if mpi_cache.rank == 0
            global e = time()
        end
    end

    #Record observables:
    if mpi_cache.rank == 0
        #Af = reshape(optimizer.mpo.A,χ,χ,2,2) 
        #Af_dagger = conj.(permutedims(Af,[1,2,4,3]))

        mx = real(tensor_calculate_magnetization(params,mpo,sx))
        my = real(tensor_calculate_magnetization(params,mpo,sy))
        mz = real(tensor_calculate_magnetization(params,mpo,sz))

        o = open("mem.out", "a")
        println(o, "k=$k: ", Base.Sys.free_memory())
        close(o)

        if mod(k,10)==1
            o = open("Ising_decay_chi$(χ)_N$(N)_hx$(hx).out", "a")
            #redirect_stdout(o)
            println(o,"k=$k: ", real(optimizer.optimizer_cache.mlL)/N, " ; acc_rate=", round(acc*100,sigdigits=2), "%", " \n M_x: ", round(mx,sigdigits=4), " \n M_y: ", round(my,sigdigits=4), " \n M_z: ", round(mz,sigdigits=4))
            println(o,Base.Sys.free_memory())
            close(o)
        end

        list_of_C = open("list_of_C_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "a")
        println(list_of_C, real(optimizer.optimizer_cache.mlL)/N)
        close(list_of_C)
        list_of_mag = open("list_of_mag_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).data", "a")
        println(list_of_mag, mx, ",", my, ",", mz)
        close(list_of_mag)

        save("MPO_density_matrix_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d).jld", "MPO_density_matrix", optimizer.mpo.A)
        #sleep(1)
        #save("MPO_density_matrix_chi$(χ)_N$(N)_hx$(hx)_γd$(γ_d)_backup.jld", "MPO_density_matrix", Af)
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

    display(mpo.A[1,:,:,:])
    display(mpo.A[2,:,:,:])
    display(mpo.A[3,:,:,:])
end