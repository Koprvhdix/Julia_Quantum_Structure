using Convex, SCS, MosekTools
using LinearAlgebra
using Random, RandomMatrices

function randState(dim)
  # d = Haar(1)
  # ru = rand(d, dim)
  # re = rand(dim)
  # ru*Diagonal(re/sum(re))*ru'
  V=randn(Complex{Float64},dim);
  V/=norm(V);
  return V*conj(transpose(V))
end

orho = [0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.25 0.25 0.   0.25 0.   0.   0.   0.25 0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.25 0.25 0.   0.25 0.   0.   0.   0.25 0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.25 0.25 0.   0.25 0.   0.   0.   0.25 0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.25 0.25 0.   0.25 0.   0.   0.   0.25 0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ;
 0.0  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  ]

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

function f_1(X_list,nps)
  p = Variable()
  rhoAs = [Semidefinite(4) for i in 1:nps]
  rhoBs = [Semidefinite(4) for i in 1:nps]
  rhoCs = [Semidefinite(4) for i in 1:nps]
  rho_next = sum(kron(X_list[1][i], rhoAs[i]) for i in 1:nps) + sum((exchange_1 * kron(X_list[2][i], rhoBs[i]) * exchange_1) for i in 1:nps) +  sum((exchange_2 * kron(X_list[3][i], rhoCs[i]) * exchange_2) for i in 1:nps)
  objective = p
  constraints = [p * orho + ((1-p) / 16) * I(16) == rho_next]
  constraints += [tr(rho) <= 1 for rho in rhoAs]
  constraints += [tr(rho) <= 1 for rho in rhoBs]
  constraints += [tr(rho) <= 1 for rho in rhoCs]
  problem = maximize(objective, constraints)
  solve!(problem, Mosek.Optimizer, silent_solver=true)
  println(problem.optval)

  # println(evaluate(rho_next))
  # p_value = evaluate(p)
  # println(p_value * orho + ((1-p_value) / 16) * I(16))

  [[rho.value for rho in rhos] for rhos in [rhoAs, rhoBs, rhoCs]], problem.optval
end

function f_2(X_list,nps)
  p = Variable()
  rhoAs = [Semidefinite(4) for i in 1:nps]
  rhoBs = [Semidefinite(4) for i in 1:nps]
  rhoCs = [Semidefinite(4) for i in 1:nps]
  rho_next = sum(kron(rhoAs[i], X_list[1][i]) for i in 1:nps) + sum((exchange_1 * kron(rhoBs[i], X_list[2][i]) * exchange_1) for i in 1:nps) +  sum((exchange_2 * kron(rhoCs[i], X_list[3][i]) * exchange_2) for i in 1:nps)
  objective = p
  constraints = [p * orho + ((1-p) / 16) * I(16) == rho_next]
  constraints += [tr(rho) <= 1 for rho in rhoAs]
  constraints += [tr(rho) <= 1 for rho in rhoBs]
  constraints += [tr(rho) <= 1 for rho in rhoCs]
  problem = maximize(objective, constraints)
  solve!(problem, Mosek.Optimizer, silent_solver=true)
  println(problem.optval)
  [[rho.value for rho in rhos] for rhos in [rhoAs, rhoBs, rhoCs]], problem.optval
end

nps = 100

for j in 1:20
    X_list = [[randState(4) for i in 1:nps] for j in 1:3]
    length = 10
    for i in 1:length
        X_list, optval = f_1(X_list,nps)
        if optval > 0.15
          length = 50
        end
        X_list, optval = f_2(X_list,nps)
        if optval > 0.15
          length = 50
        end
    end
    println("one round ", j)
end
