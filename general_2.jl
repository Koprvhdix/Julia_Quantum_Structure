using Polyhedra
import GLPK

using Convex, SCS

using LinearAlgebra

lib = DefaultLibrary{Float64}(GLPK.Optimizer)

exchange_1 = [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 ]

exchange_2 = [ 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 ;
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]

poly_stable = polyhedron(vrep([vec(Matrix(I, 16,  16) / 16)]), lib)
global poly_stable

for i in 1:45
  point_list = Vector{Float64}[]
    point_1_1 = normalize(rand(1, 4))
    point_1_2 = normalize(rand(1, 4))

    point_2_1 = normalize(rand(1, 4))
    point_2_2 = normalize(rand(1, 4))

    point_3_1 = normalize(rand(1, 4))
    point_3_2 = normalize(rand(1, 4))

    point_1 = kron(point_1_1, point_1_2)
    point_1 = point_1' * point_1
    push!(point_list, vec(point_1))

    point_2 = kron(point_2_1, point_2_2)
    point_2 = exchange_1 * (point_2' * point_2) * exchange_1
    push!(point_list, vec(point_2))

    point_3 = kron(point_3_1, point_3_2)
    point_3 = exchange_2 * (point_3' * point_3) * exchange_2
    push!(point_list, vec(point_3))

  global poly_stable = convexhull(poly_stable, vrep(point_list))
  h = hrep(poly_stable)  
  println(i)
end

result_list = [10.0]

h = hrep(poly_stable)
# global count = 1
for pl in eachindex(halfspaces(h))
    w = (get(h, pl).a)'
    b = get(h, pl).β

    if dot(w, vec(I(16))) - b >= 0.000000001
      continue
    end

    rho = Semidefinite(16)

    objective = dot(w, vec(rho)) 
    # objective = dot(w, vec(I(16) / 16)) / 2 / (-dot(w, vec(rho)) + dot(w, vec(I(16) / 16)) / 2)
    # objective = (dot(w, vec(I(16))) - b) / (dot(w, vec(I(16))) - 16 * dot(w, vec(rho)))

    constraints = [tr(rho) == 1, rho in :SDP]

    problem = maximize(objective, constraints)

    solve!(problem, SCS.Optimizer)

    result = (dot(w, vec(I(16))) - b) / (dot(w, vec(I(16))) - 16 * dot(w, vec(evaluate(rho))))
    
    # result_rho = evaluate(rho)
    # println(result_rho)
    # println(tr(result_rho))
    # println(eigvals(result_rho))

    # push!(result_list, problem.optval)
    if result < 1 && result > 0
      push!(result_list, result)
    end
end

println(result_list)
println(minimum(result_list))
