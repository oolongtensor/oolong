using TensorDSL
using BenchmarkTools
using PyCall

dim = 40
V = VectorSpace(dim)
A = VariableTensor("A", V, V', V)
x = FreeIndex(V, "x")
knl = Kernel((4*A)[x, x'])

b = @benchmarkable execute($knl, "A"=>val) setup = (val = rand(Float64, (dim, dim, dim))) evals = 1
run(b, samples=100)