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

exchange_2 = [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 ;
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 ]

base_point_list = [ vec([cos(i)*cos(j) cos(i)*sin(j) sin(i)*cos(k) sin(i)*sin(k)]) for i in range(-pi, stop=pi, length=5) for j in range(-pi, stop=pi, length=5) for k in range(-pi, stop=pi, length=5) ]

current_point = kron(base_point_list[1], base_point_list[1])
current_point_1 = current_point * current_point'

point_list = [ vec(current_point_1) ]

for point_1 in base_point_list
  for point_2 in base_point_list
    current_point = kron(point_1, point_2)

    current_point_1 = current_point * current_point'
    current_point_2 = exchange_1 * (current_point * current_point') * exchange_1
    current_point_3 = exchange_2 * (current_point * current_point') * exchange_2
    push!(point_list, vec(current_point_1))
    push!(point_list, vec(current_point_2))
    push!(point_list, vec(current_point_3))
  end
end

# for i in 1:1000
#     point_1_1 = normalize(rand(1, 4))
#     point_1_2 = normalize(rand(1, 4))

#     point_2_1 = normalize(rand(1, 4))
#     point_2_2 = normalize(rand(1, 4))

#     point_3_1 = normalize(rand(1, 4))
#     point_3_2 = normalize(rand(1, 4))

#     point_1 = kron(point_1_1, point_1_2)
#     point_1 = point_1' * point_1
#     push!(point_list, vec(point_1))

#     point_2 = kron(point_2_1, point_2_2)
#     point_2 = exchange_1 * (point_2' * point_2) * exchange_1
#     push!(point_list, vec(point_2))

#     point_3 = kron(point_3_1, point_3_2)
#     point_3 = exchange_2 * (point_3' * point_3) * exchange_2
#     push!(point_list, vec(point_3))
# end

println("End get point")

poly = polyhedron(vrep(point_list), lib)
removevredundancy!(poly)

println("End get poly")

h = hrep(poly)

println("End get hrep")

# # SDP

# result_list = [0.0]

# println(vec(I(16) / 16))

# for hs in eachindex(halfspaces(h))
#   w = (get(h, hs).a)'
#   b = get(h, hs).β
# for pl in eachindex(hyperplanes(h))
#     w = (get(h, pl).a)'
#     b = get(h, pl).β
#   # println(w)
#   println("B")
#   println(b)

#   result = dot(w, vec(I(16) / 16))
#   println(result)

#   println("Point")
#   for point in point_list
#     println(dot(w, point) - b)
#   end
# end


# for pl in eachindex(hyperplanes(h))
#     w = (get(h, pl).a)'
#     b = get(h, pl).β

#     rho = Semidefinite(16)

#     objective = -(dot(w, vec(rho)) - b)

#     constraints = [tr(rho) == 1, rho in :SDP]

#     problem = minimize(objective, constraints)

#     solve!(problem, SCS.Optimizer)

#     result = dot(w, vec(I(16) / 16)) / 2 / (-dot(w, vec(evaluate(rho))) + dot(w, vec(I(16) / 16)) / 2)

#     push!(result_list, result)
# end

# for hs in eachindex(halfspaces(h))
#   w = (get(h, hs).a)'
#   b = get(h, hs).β

#   rho = Semidefinite(16)

#   objective = -(dot(w, vec(rho)) - b)

#   constraints = [tr(rho) == 1, rho in :SDP]

#   problem = minimize(objective, constraints)

#   solve!(problem, SCS.Optimizer)

#   result = dot(w, vec(I(16) / 16)) / (-dot(w, vec(evaluate(rho))) + dot(w, vec(I(16) / 16)))

#   push!(result_list, result)
# end

# for number in result_list
#     println(number)
# end
