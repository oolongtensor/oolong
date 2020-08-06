module TensorDSL

export
Node,

AbstractVectorSpace, VectorSpace, DualVectorSpace, RnSpace, dual, dim,

Index, FreeIndex, FixedIndex, Indices, toindex,

AbstractTensor, TerminalTensor, ScalarVariable, Scalar, VariableTensor,
Tensor, DeltaTensor, ZeroTensor, ConstantTensor,

Operation, IndexSumOperation, AddOperation, OuterProductOperation, ⊗,
IndexingOperation, ComponentTensorOperation, componenttensor,

SineOperation, sin, CosineOperation, cos, TangentOperation, tan,

diff,

Assignment, assign,

updatechildren, updatevectorspace

include("TensorPrototype/Assignment.jl")
include("TensorPrototype/Indices.jl")
include("TensorPrototype/Node.jl")
include("TensorPrototype/Operations.jl")
include("TensorPrototype/Tensors.jl")
include("TensorPrototype/Trigonometry.jl")
include("TensorPrototype/VectorSpace.jl")
include("TensorPrototype/Differentation.jl")
include("TreeVisitor/UpdateNodes.jl")
include("TreeVisitor/Traversal.jl")

end
