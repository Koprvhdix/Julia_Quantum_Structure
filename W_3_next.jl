using Convex, SCS, MosekTools
using LinearAlgebra
using Random, RandomMatrices

# setprecision(20)

function randState(d::Int=2, L::Int=100)
  QPolytope= Array{Array{Complex{Float64},2},1}(undef,L);
  for k in 1:L
      V=randn(Complex{Float64},d);
      V/=norm(V);
      QPolytope[k]=V*conj(transpose(V));
  end
  return QPolytope;
end

struct MultiState
	mat::Matrix{Complex{Float64}}
	dims::Array{Int,1}
end;

function AccumSizes(dims::Array{Int,1})
  n= length(dims);
sizes=Array{Int,1}(undef,n);
s=1;
  for k in reverse(1:n)
  s*= dims[k];
  sizes[k]=s;
  end;
return sizes;
end;

function LocalIndex(L::Int,dd::Array{Int,1})
	n=length(dd);
	sizes= AccumSizes(dd);
	kk=Array{Int,1}(undef,n);
	s=L-1;
	for k in reverse(1:n)
		kk[k]= mod(s,dd[k])+1;
		s= div(s,dd[k]);
	end;
	#kA=div(L-1,dd[2])+1;
	#kB=mod(L-1,dd[2])+1;
	return kk;
end;

function LinearIndex(kk::Array{Int,1},dd::Array{Int,1})
	sizes= AccumSizes(dd);
	n= length(kk);
	s=kk[n]-1;
	for k in reverse(1:(n-1))
		s+= sizes[k+1]*(kk[k]-1);
	end;
	return s+1;
end;

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

my_rho = MultiState(orho, [2, 2, 2])

function f_1(X_list,nps)
    p = Variable()
    rhoAs = [HermitianSemidefinite(2) for i in 1:nps]
    rhoBs = [HermitianSemidefinite(2) for i in 1:nps]
    rhoCs = [HermitianSemidefinite(2) for i in 1:nps]
    rho_next = sum(kron(rhoAs[i], X_list[i]) for i in 1:nps) + sum((exchange * kron(X_list[i], rhoBs[i]) * exchange) for i in 1:nps) +  sum(kron(X_list[i], rhoCs[i]) for i in 1:nps)
    objective = p
    constraints = [p * orho + ((1-p) / 8) * I(8) == rho_next]
    constraints += [tr(rho) <= 1 for rho in rhoAs]
    constraints += [tr(rho) <= 1 for rho in rhoBs]
    constraints += [tr(rho) <= 1 for rho in rhoCs]
    problem = maximize(objective, constraints)
    solve!(problem, Mosek.Optimizer, silent_solver=true)
    println(problem.optval)
    [[rho.value for rho in rhos] for rhos in [rhoAs, rhoBs, rhoCs]], problem.optval
end

function f_2(rho, X_list,nps)
    # p=Variable(1,Positive());
    # rhoAs = [HermitianSemidefinite(4,4) for i in 1:nps]
    # rhoBs = [HermitianSemidefinite(4,4) for i in 1:nps]
    # rhoCs = [HermitianSemidefinite(4,4) for i in 1:nps]
    # # rho_next = sum(kron(X_list[i], rhoAs[i]) for i in 1:nps) +  sum((exchange * kron(X_list[i], rhoBs[i]) * exchange) for i in 1:nps) +  sum(kron(rhoCs[i], X_list[i]) for i in 1:nps)
    
    # global rho_next=zeros(Complex{Float64},prod(rho.dims),prod(rho.dims));
    # for i in 1:nps
    #   global rho_next+= kron(X_list[i], rhoAs[i]);
    #   global rho_next+= kron(rhoCs[i], X_list[i]);
      
    #   C= zeros(Complex{Float64},8,8);
    #   for l in 1:8
    #   kk= LocalIndex(l,[2,2,2]); tmp=kk[1]; kk[1]=kk[2]; kk[2]= tmp;
    #   p=LinearIndex(kk,[2,2,2]);
    #     C[p,l]=1.0;
    #   end;
    #   global rho_next += C*kron(X_list[i], rhoBs[i])*C; 
    # end;
    
    # II= zeros(Complex{Float64},prod(rho.dims),prod(rho.dims))+I;
    # problem = maximize(p)
    # problem.constraints += ((p*rho.mat+(1.0-p)/prod(rho.dims)*II) == rho_next)
    # solve!(problem, Mosek.Optimizer)
    # println(problem.optval)
    # rho = MultiState(orho, [2, 2, 2]);
  t=Variable(1,Positive());
  rhoAs = [HermitianSemidefinite(4,4) for i in 1:nps]
  rhoBs = [HermitianSemidefinite(4,4) for i in 1:nps]
  rhoCs = [HermitianSemidefinite(4,4) for i in 1:nps]

  global C= zeros(Complex{Float64},8,8);
  for l in 1:8
    kk= LocalIndex(l,[2,2,2]); tmp=kk[1]; kk[1]=kk[2]; kk[2]= tmp;
    p=LinearIndex(kk,[2,2,2]);
    C[p,l]=1.0;
  end;

  # rho_next = sum(kron(X_list[i], rhoAs[i]) for i in 1:nps) +  sum((C * kron(X_list[i], rhoBs[i]) * C) for i in 1:nps) +  sum(kron(rhoCs[i], X_list[i]) for i in 1:nps)

  global rho_next=zeros(Complex{Float64},prod(rho.dims),prod(rho.dims));
  for i in 1:nps
    rho_next+= kron(X_list[i], rhoAs[i]);
    rho_next+= kron(rhoCs[i], X_list[i]);
    
    # global C= zeros(Complex{Float64},8,8);
    # for l in 1:8
    # kk= LocalIndex(l,[2,2,2]); tmp=kk[1]; kk[1]=kk[2]; kk[2]= tmp;
    # p=LinearIndex(kk,[2,2,2]);
    #   C[p,l]=1.0;
    # end;
    rho_next += C*kron(X_list[i], rhoBs[i])*C; 
  end;

  II= zeros(Complex{Float64},prod(rho.dims),prod(rho.dims))+I;
  problem= maximize(t);
  problem.constraints+= ((p*rho.mat+(1.0-t)/prod(rho.dims)*II) == rho_next);
  solve!(problem, Mosek.Optimizer(LOG=0));
  print(problem.status,"\n");
  print("robustness to white noise: ",problem.optval, "\n"); 
end

# X_list = randState(2)
# println(X_list[1])

nps = 200
# # X_list = [[randState(4) for i in 1:nps] for j in 1:3]
# for j in 1:10
#     X_list = randState(2)
#     for i in 1:length
#       X_list, optval = f_2(X_list,nps)
#       X_list, optval = f_1(X_list,nps)
#     end
#     println("one round ", j)
# end

X_list = randState(2, 200);
# f_2(my_rho, X_list, nps)

rho = MultiState(orho, [2, 2, 2]);
t=Variable(1,Positive());
rhoAs = [HermitianSemidefinite(4,4) for i in 1:nps]
# rhoBs = [HermitianSemidefinite(4,4) for i in 1:nps]
# rhoCs = [HermitianSemidefinite(4,4) for i in 1:nps]

global C= zeros(Complex{Float64},8,8);
for l in 1:8
  kk= LocalIndex(l,[2,2,2]); tmp=kk[1]; kk[1]=kk[2]; kk[2]= tmp;
  p=LinearIndex(kk,[2,2,2]);
  C[p,l]=1.0;
end;

rho_next = sum(kron(X_list[i], rhoAs[i]) for i in 1:nps) +  sum((C*kron(X_list[i], rhoAs[i])*C) for i in 1:nps) +  sum(kron(rhoAs[i], X_list[i]) for i in 1:nps)

# global rho_next=zeros(Complex{Float64},prod(rho.dims),prod(rho.dims));
# for i in 1:nps
#   global rho_next+= kron(X_list[i], rhoAs[i]);
#   global rho_next+= kron(rhoCs[i], X_list[i]);
#   C= zeros(Complex{Float64},8,8);
#   for l in 1:8
#   kk= LocalIndex(l,[2,2,2]); tmp=kk[1]; kk[1]=kk[2]; kk[2]= tmp;
#   p=LinearIndex(kk,[2,2,2]);
#     C[p,l]=1.0;
#   end;
#   global rho_next += C*kron(X_list[i], rhoAs[i])*C; 
# end;

II= zeros(Complex{Float64},prod(rho.dims),prod(rho.dims))+I;
problem= maximize(t);
problem.constraints+= ((t*rho.mat+(1.0-t)/prod(rho.dims)*II) == rho_next);
solve!(problem, Mosek.Optimizer(LOG=0));
print(problem.status,"\n");
print("robustness to white noise: ",problem.optval, "\n"); 

function nlize(rho)
  evs = eigvals(rho)
  for it in evs
    if real(it) < 0 || imag(it) != 0
      println("evs ", it)
    end
  end

  for i in 1:size(rho, 1)
    if imag(rho[i, i]) != 0 || real(rho[i, i]) < 0
      println("diag ", rho[i, i])
    end
  end
end

for rho in rhoAs
  nlize(rho.value)
end
# next_rho = [nlize(rho) for rho in rhoBs]
# next_rho = [nlize(rho) for rho in rhoCs]

# II= zeros(Complex{Float64}, 8, 8)+I;
# II= zeros(Complex{Float64},prod(rho.dims),prod(rho.dims))+I;
# # # constraints = [p * orho + ((1-p) / 8) * II == rho_next]
# problem = maximize(p)
# problem.constraints += ((p*rho.mat+(1.0-p)/prod(rho.dims)*II) == rho_next)
# solve!(problem, Mosek.Optimizer(LOG=0), silent_solver=true)
# println(problem.optval)
# [[rho.value for rho in rhos] for rhos in [rhoAs, rhoBs, rhoCs]], problem.optval
