using Convex, SCS, MosekTools
using LinearAlgebra
using Random, RandomMatrices

using SDPAFamily

setprecision(20)

opt = () -> SDPAFamily.Optimizer{BigFloat}(presolve = true)

function randState(d::Int=2, L::Int=100)
  QPolytope= Array{Array{Complex{BigFloat},2},1}(undef,L);
  for k in 1:L
      V=randn(Complex{BigFloat},d);
      V/=norm(V);
      QPolytope[k]=V*conj(transpose(V));
  end
  return QPolytope;
end

struct MultiState
	mat::Matrix{Complex{BigFloat}}
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

orho = Complex{BigFloat}[0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   1/3  1/3  0.   1/3  0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ;
      0.   0.   0.   0.   0.   0.   0.   0.  ]

exchange = Complex{BigFloat}[1. 0. 0. 0. 0. 0. 0. 0. ;
      0. 0. 1. 0. 0. 0. 0. 0. ;
      0. 1. 0. 0. 0. 0. 0. 0. ;
      0. 0. 0. 1. 0. 0. 0. 0. ;
      0. 0. 0. 0. 1. 0. 0. 0. ;
      0. 0. 0. 0. 0. 0. 1. 0. ;
      0. 0. 0. 0. 0. 1. 0. 0. ;
      0. 0. 0. 0. 0. 0. 0. 1. ]

my_rho = MultiState(orho, [2, 2, 2])

nps = 5


X_list = randState(2, nps);


rho = MultiState(orho, [2, 2, 2]);
t=Variable(1,Positive());
rhoAs = [HermitianSemidefinite(4,4) for i in 1:nps]

global C= zeros(Complex{BigFloat},8,8);
for l in 1:8
  kk= LocalIndex(l,[2,2,2]); tmp=kk[1]; kk[1]=kk[2]; kk[2]= tmp;
  p=LinearIndex(kk,[2,2,2]);
  C[p,l]=1.0;
end;

rho_next = sum(kron(X_list[i], rhoAs[i]) for i in 1:nps) +  sum((C*kron(X_list[i], rhoAs[i])*C) for i in 1:nps) +  sum(kron(rhoAs[i], X_list[i]) for i in 1:nps)

II= zeros(Complex{BigFloat},prod(rho.dims),prod(rho.dims))+I;
problem= maximize(t, [(t*rho.mat+(1.0-t)/prod(rho.dims)*II) == rho_next]; numeric_type = BigFloat);
solve!(problem, opt);
print(problem.status,"\n");
print("robustness to white noise: ",problem.optval, "\n"); 

