include("../Trigonometry.jl")
include("../Differentation.jl")
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
Z = ZeroTensor(V3, V2)
a = ScalarVariable("a")
aTensor = Tensor([a])

@testset "Operations" begin
    @testset "Addition" begin
        @test (A + B).shape == A.shape
        @test A + Z == A
        @test (A + Z)[1, 2] == A[1, 2]
        @test Z + Z + Z == Z
        @test (A + B).children == (A, B)
        @test (A[x,1] + B[1, y]).freeindices == (x, y)
        @test_throws DimensionMismatch A + D
        @test (A[x, y] + D[y', z]).freeindices == (x,z)
    end
    @testset "Index" begin
        @test A[x, 1].shape == ()
        @test A[x, 1].freeindices == (x,)
        @test_throws DimensionMismatch A[x, x]
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
        @test componenttensor(A[x, 1], x).shape == (V3,)
        @test componenttensor(A[x, y], y).shape == (V2,)
        @test componenttensor(A[x, y], y, x).shape == (V2, V3)
        @test componenttensor(E[y', x', z], x', z).shape == (V3', Vi)
        @test componenttensor(E[y', x', z], x', z).freeindices == (y',)
        @test_throws DomainError componenttensor(A[x, y], x, z)
    end
    @testset "Tensor contraction" begin
        @test (A[x, y]*D[y', z]).shape == ()
        @test (A[x, y]*D[y', z]).freeindices == (x, z)
        @test (A[x, y]*E[y', x', z]).shape == ()
        @test (A[x, y]*E[y', x', z]).freeindices == (z,)
        @test componenttensor(C[w, z, z'], w).shape == (Vj,)
        @test C[w, z, z'].freeindices == (w,)
        @test C[w, z, z'].children[2] == Indices(z)
    end
end
@testset "Tensors" begin
    @test_throws DomainError Tensor([1, 2], VectorSpace(3), VectorSpace(2))
    @test_throws DomainError Tensor([1, 2], VectorSpace(3))
end
@testset "Indices" begin
    @test_throws DomainError FixedIndex(VectorSpace(4), 5)
    @test_throws DomainError FixedIndex(VectorSpace(), 5)
end
@testset "Trigonometry" begin
    @test sin(1) isa SineOperation
    @test sin(3).children[1].value == Tensor(3).value
    @test cos(ScalarVariable("x")) isa CosineOperation
    @test tan(B[1, 2]) isa TangentOperation
    @test_throws MethodError sin(A)
    @test_throws MethodError cos(A)
    @test_throws MethodError tan(A)
end
@testset "Differentation" begin
    @test diff(cos(ScalarVariable("x")), ScalarVariable("x")) isa DifferentationOperation
    @test diff(cos(aTensor), a).children == (cos(aTensor), a)
    @test differentiateAST(diff(aTensor, a)) == DeltaTensor()
    @test differentiateAST(aTensor) == aTensor
    @test differentiateAST(diff(aTensor, ScalarVariable("z"))) == ZeroTensor()
    @test differentiateAST(diff(aTensor + ZeroTensor(), a)) == DeltaTensor()
end