using Convex, SCS, MosekTools
using LinearAlgebra
using Random, RandomMatrices

function randState(dim, nps)
    re = randn(dim)+ [i*im for i in randn(dim)]
    V = re*re'/(re'*re)
    return V / nps
end

# the W state
orho = [0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ]


function f_1(X_list,nps) # fix BC in A|BC, AC in B|AC and AB in AB|C, then optimize over others.
  global exchange = [1. 0. 0. 0. 0. 0. 0. 0. ;
              0. 0. 1. 0. 0. 0. 0. 0. ;
              0. 1. 0. 0. 0. 0. 0. 0. ;
              0. 0. 0. 1. 0. 0. 0. 0. ;
              0. 0. 0. 0. 1. 0. 0. 0. ;
              0. 0. 0. 0. 0. 0. 1. 0. ;
              0. 0. 0. 0. 0. 1. 0. 0. ;
              0. 0. 0. 0. 0. 0. 0. 1. ];
    p = Variable()
    rhoAs = [HermitianSemidefinite(2) for i in 1:nps]
    rhoBs = [HermitianSemidefinite(2) for i in 1:nps]
    rhoCs = [HermitianSemidefinite(2) for i in 1:nps]
    rho_next = sum(kron(rhoAs[i], X_list[1][i]) for i in 1:nps) + sum((exchange * kron(X_list[2][i], rhoBs[i]) * exchange) for i in 1:nps) +  sum(kron(X_list[3][i], rhoCs[i]) for i in 1:nps)
    objective = p
    constraints = [p * orho + ((1-p) / 8) * I(8) == rho_next, p>=0, p<=1]
    problem = maximize(objective, constraints)
    #solve!(problem, Mosek.Optimizer, silent_solver=true, verbose=false)
    solve!(problem, Mosek.Optimizer(LOG=0), silent_solver=true, verbose=false)
    [problem.optval,[[rho.value for rho in rhos] for rhos in [rhoAs, rhoBs, rhoCs]]]
end

function f_2(X_list,nps) # fix A in A|BC, B in B|AC and C in AB|C, then optimize over others
  global exchange = [1. 0. 0. 0. 0. 0. 0. 0. ;
              0. 0. 1. 0. 0. 0. 0. 0. ;
              0. 1. 0. 0. 0. 0. 0. 0. ;
              0. 0. 0. 1. 0. 0. 0. 0. ;
              0. 0. 0. 0. 1. 0. 0. 0. ;
              0. 0. 0. 0. 0. 0. 1. 0. ;
              0. 0. 0. 0. 0. 1. 0. 0. ;
              0. 0. 0. 0. 0. 0. 0. 1. ];
    p = Variable(1,Positive());
    rhoAs = [HermitianSemidefinite(4) for i in 1:nps]
    # rhoBs = [HermitianSemidefinite(4) for i in 1:nps]
    # rhoCs = [HermitianSemidefinite(4) for i in 1:nps]
    rho_next = sum(kron(X_list[1][i], rhoAs[i]) for i in 1:nps) + sum((exchange * kron(rhoAs[i], X_list[2][i]) * exchange) for i in 1:nps) +  sum(kron(rhoAs[i], X_list[3][i]) for i in 1:nps)
    objective = p
    constraints = [p * orho + ((1.0-p) / 8.) * I(8) == rho_next]
    problem = maximize(objective, constraints)
    solve!(problem, Mosek.Optimizer(LOG=0), silent_solver=true, verbose=false)
    println(problem.optval)
    # [problem.optval,[[rho.value for rho in rhos] for rhos in [rhoAs, rhoBs, rhoCs]]]
    [problem.optval,[rho.value for rho in rhoAs]]
end

# A test
id2 = Matrix(I,2,2)
nps = 50
res = []
Xs = []
pX_list = push!([randState(2, nps) for i in 1:nps])
X_list = [pX_list for j in 1:3]
pvalue, X_list = f_2(X_list, nps)
println(pvalue)

global error_count = 0
for current_matrix in X_list
    evs = eigvals(current_matrix)
    revs = [real(it) for it in evs]
    ievs = [imag(it) for it in evs]
    if minimum(revs) < 0
      println(revs)
      global error_count += 1
    else

    end
end

println(" error count ", error_count)

# for k in 1:20 # 10 random starting point
#     pX_list = push!([randState(2) for i in 1:nps-1],id2/2)
#     X_list = [pX_list for j in 1:3]
#     for i in 1:3 
#         push!(Xs,X_list)
#         pvalue, X_list = f_2(X_list,nps)
#         #X_list = [[(rho+rho')/tr(rho+rho') for rho in rhos] for rhos in X_list]
#         X_list = [[rho/tr(rho) for rho in rhos] for rhos in X_list]
#         test = [eigvals(mat) for mat in X_list[1]]
#         print(test)
#         push!(res,pvalue)
#         push!(Xs,X_list)
#         pvalue,X_list = f_1(X_list,nps)
#         X_list = [[rho/tr(rho) for rho in rhos] for rhos in X_list]
#         push!(res,pvalue)
#     end
#     println(maximum(res))
#     if maximum(res)>0.5
#       test = [eigvals(mat) for mat in X_list[1]]
#       print(test)
#       break
#     end
#     println(k)
# end
