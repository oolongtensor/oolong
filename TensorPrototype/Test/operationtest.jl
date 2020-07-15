include("../Operations.jl")
using Test

V3 = VectorSpace(3)
V2 = VectorSpace(2)
Vi = VectorSpace()
Vj = VectorSpace()

x = FreeIndex(V3, "x")
y = FreeIndex(V2, "y")
z = FreeIndex(Vi, "z")
w = FreeIndex(Vj, "w")

A = VariableTensor(V3, V2)
B = Tensor(fill(1.2, (3, 2)), V3, V2)
C = VariableTensor(Vj, Vi, Vi')
D = VariableTensor(V2', Vi)
E = VariableTensor(V2', V3', Vi)
@testset "Operations" begin
    @testset "Addition" begin
        @test (A + B).shape == A.shape
        @test (A + B).children == (A, B)
        @test (A[x,1] + B[1, y]).freeindices == (x, y)
        @test_throws ErrorException A[x, y] + B
    end
    @testset "Index" begin
        @test A[x, 1].shape == ()
        @test A[x, 1].freeindices == (x,)
        @test_throws ErrorException A[x, x]
        @test A[x].shape == (V2,)
        @test A[x] isa ComponentTensorOperation
        @test E[y'].shape == (V3', Vi)
    end
    @testset "Outer product" begin
        @test (A⊗B).shape == (V3, V2, V3, V2)
        @test (A⊗B).children == (A, B)
        @test (A[x,1] ⊗ B[1, y]).freeindices == (x, y)
        @test (A - B).shape == (V3, V2)
        @test (-B).shape == (V3, V2)
        @test (A - B) isa AddOperation
    end
    @testset "Component tensor" begin
        @test componentTensor(A[x, 1], x).shape == (V3,)
        @test componentTensor(A[x, y], y).shape == (V2,)
        @test componentTensor(A[x, y], y, x).shape == (V2, V3)
        @test componentTensor(E[y', x', z], x', z).shape == (V3', Vi)
        @test componentTensor(E[y', x', z], x', z).freeindices == (y',)
    end
    @testset "Index sum" begin
        @test indexsum(C[w, z, z'], z).shape == ()
        @test indexsum(C[w, z, z'], z).freeindices == (w,)
        @test indexsum(C[w, z, z'], z).children[2] == Indices(z)
    end
    @testset "Tensor contraction" begin
        @test (A[x, y]*D[y', z]).shape == (V3, Vi)
        @test (A[x, y]*D[y', z]).freeindices == Set()
        @test (A[x, y]*E[y', x', z]).shape == (Vi,)
        @test (A[x, y]*E[y', x', z]).freeindices == Set()
        @test tensorcontraction(C[w, z, z']).shape == (Vj,)
    end
    @testset "Adjacent indices" begin
        @test getadjacentindices(x, x', y, z) == (x,)
        @test getadjacentindices(x, y, y', x') == (y, x)
        @test length(getadjacentindices(x, y,z, y', x')) == 0
        @test getadjacentindices(x, y, y', x', z, z') == (y, x, z)
    end
end
