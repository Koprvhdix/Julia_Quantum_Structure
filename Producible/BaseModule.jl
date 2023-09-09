module BaseModule
  struct MultiState
    mat::Matrix{Complex{Float64}}
    dims::Array{Int,1}
  end;
  export MultiState

  function randState(dim)
    V=randn(Complex{Float64},dim);
    V/=norm(V);
    return V*conj(transpose(V))
  end;
  export randState

  function nlize(rho)
    evs = eigvals(rho)
    revs = [real(it) for it in evs]
    ievs = [imag(it) for it in evs]
    if ievs'*ievs/(revs'*revs)> 1e-3 || minimum(revs) < 0
      decompose_and_reconstruct_positive(rho)
    else rho/tr(rho)
    end
  end;
  export nlize
end
