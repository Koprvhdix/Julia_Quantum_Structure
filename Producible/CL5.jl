include("BaseModule.jl")

using Convex, SCS, MosekTools
using LinearAlgebra
using Random, RandomMatrices

using .BaseModule

Cl5_matrix = zeros(32, 32)

indices = [1, 16, 20, 29]
for index in indices
    for index2 in indices
        Cl5_matrix[index, index2] = 0.25
    end
end

print(Cl5_matrix)


