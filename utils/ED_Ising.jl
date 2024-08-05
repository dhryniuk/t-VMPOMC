export generate_bit_basis_reversed

#Ising bit-basis:
function generate_bit_basis_reversed(N)#(N::UInt8)
    set::Vector{Vector{Bool}} = [[false], [true]]
    @simd for i in 1:N-1
        new_set::Vector{Vector{Bool}} = []
        @simd for state in set
            state2::Vector{Bool} = copy(state)
            state = vcat(state, false)
            state2 = vcat(state2, true)
            push!(new_set, state)
            push!(new_set, state2)
        end
        set = new_set
    end
    return Vector{Vector{Bool}}(set)
end