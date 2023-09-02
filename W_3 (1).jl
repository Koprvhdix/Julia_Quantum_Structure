using Convex, SCS, MosekTools
using LinearAlgebra
using Random, RandomMatrices

function randState(dim)
    d = Haar(1)
    ru = rand(d, dim)
    re = rand(dim)
    ru*Diagonal(re/sum(re))*ru'
end

# nps = 20

orho = [0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ]

exchange = [1. 0. 0. 0. 0. 0. 0. 0. ;
      0. 0. 1. 0. 0. 0. 0. 0. ;
      0. 1. 0. 0. 0. 0. 0. 0. ;
      0. 0. 0. 1. 0. 0. 0. 0. ;
      0. 0. 0. 0. 1. 0. 0. 0. ;
      0. 0. 0. 0. 0. 0. 1. 0. ;
      0. 0. 0. 0. 0. 1. 0. 0. ;
      0. 0. 0. 0. 0. 0. 0. 1. ]

function f_1(X_list,nps)
    p = Variable()
    rhoAs = [Semidefinite(2) for i in 1:nps]
    rhoBs = [Semidefinite(2) for i in 1:nps]
    rhoCs = [Semidefinite(2) for i in 1:nps]
    rho_next = sum(kron(rhoAs[i], X_list[1][i]) for i in 1:nps) + sum((exchange * kron(X_list[2][i], rhoBs[i]) * exchange) for i in 1:nps) +  sum(kron(X_list[3][i], rhoCs[i]) for i in 1:nps)
    objective = p
    constraints = [p * orho + ((1-p) / 8) * I(8) == rho_next]
    constraints += [tr(rho) <= 1 for rho in rhoAs]
    constraints += [tr(rho) <= 1 for rho in rhoBs]
    constraints += [tr(rho) <= 1 for rho in rhoCs]
    problem = maximize(objective, constraints)
    solve!(problem, SCS.Optimizer, silent_solver=true)
    println(-1*problem.optval)
    [[rho.value for rho in rhos] for rhos in [rhoAs, rhoBs, rhoCs]]
end

function f_2(X_list,nps)
    p = Variable()
    rhoAs = [Semidefinite(4) for i in 1:nps]
    rhoBs = [Semidefinite(4) for i in 1:nps]
    rhoCs = [Semidefinite(4) for i in 1:nps]
    rho_next = sum(kron(X_list[1][i], rhoAs[i]) for i in 1:nps) + sum((exchange * kron(rhoBs[i], X_list[2][i]) * exchange) for i in 1:nps) +  sum(kron(rhoCs[i], X_list[3][i]) for i in 1:nps)
    objective = p
    constraints = [p * orho + ((1-p) / 8) * I(8) == rho_next]
    constraints += [tr(rho) <= 1 for rho in rhoAs]
    constraints += [tr(rho) <= 1 for rho in rhoBs]
    constraints += [tr(rho) <= 1 for rho in rhoCs]
    problem = maximize(objective, constraints)
    solve!(problem, SCS.Optimizer, silent_solver=true)
    println(-1*problem.optval)
    [[rho.value for rho in rhos] for rhos in [rhoAs, rhoBs, rhoCs]]
end

nps = 200
# X_list = [[randState(4) for i in 1:nps] for j in 1:3]
for j in 1:10
    X_list = [[randState(4) for i in 1:nps] for j in 1:3]
    for i in 1:10
        X_list = f_1(X_list,nps)
        X_list = f_2(X_list,nps)
    end
    println("one round")
end
