var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"Hello World!","category":"page"},{"location":"operations/","page":"Operations","title":"Operations","text":"Base.:+(nodes::Vararg{Node})\n⊗(A::AbstractTensor, B::AbstractTensor)\nBase.:*(x::Scalar, A::AbstractTensor)\nBase.:-(A::AbstractTensor)\nBase.:-(A::AbstractTensor{rank}, B::AbstractTensor{rank}) where rank\nBase.:^(x::AbstractTensor{0}, y::AbstractTensor{0})\nBase.:^(x::AbstractTensor{0}, y::Number)\nBase.sqrt(x::AbstractTensor{0})\nBase.:/(A::AbstractTensor, y::AbstractTensor{0})\nBase.:/(y::Number, A::AbstractTensor{0})\ncomponenttensor(A::AbstractTensor, indices::Vararg{Index})\nBase.getindex(A::AbstractTensor, ys::Vararg{Index})\nBase.getindex(A::AbstractTensor, ys::Vararg{Union{String, Int, Index}})\nBase.transpose(A::AbstractTensor)\ntrace(A::AbstractTensor{2})","category":"page"},{"location":"operations/#Base.:+-Tuple{Vararg{Node,N} where N}","page":"Operations","title":"Base.:+","text":"Base.:+(nodes::Vararg{Node})\n\nCreates an addoperation whose children are the nodes.\n\n\n\n\n\n","category":"method"},{"location":"operations/#TensorDSL.:⊗-Tuple{AbstractTensor,AbstractTensor}","page":"Operations","title":"TensorDSL.:⊗","text":"⊗(A::AbstractTensor, B::AbstractTensor)\n\nReturns an OuterProductOperation with A and B as its children.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.:*-Tuple{Union{Real, AbstractTensor{0}, Complex},AbstractTensor}","page":"Operations","title":"Base.:*","text":"Base.:*(x::Scalar, A::AbstractTensor)\n\nShorthand for multiplying a tensor by a scalar.  If multiplying a number by a variable, the number must be first.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.:--Tuple{AbstractTensor}","page":"Operations","title":"Base.:-","text":"Base.:-(A::AbstractTensor)\n\nUnary minus. Creates an OuterProductOperation between ConstantTensor(-1) and A.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.:--Union{Tuple{rank}, Tuple{AbstractTensor{rank},AbstractTensor{rank}}} where rank","page":"Operations","title":"Base.:-","text":"Base.:-(A::AbstractTensor{rank}, B::AbstractTensor{rank}) where rank\n\nBinary minus. Creates an AddOperation of A and -1*B.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.:^-Tuple{AbstractTensor{0},AbstractTensor{0}}","page":"Operations","title":"Base.:^","text":"Base.:^(x::AbstractTensor{0}, y::AbstractTensor{0})\n\nRaises the scalar-shaped tensor x to the scalar-shaped power y. Returns a PowerOperation.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.:^-Tuple{AbstractTensor{0},Number}","page":"Operations","title":"Base.:^","text":"Base.:^(x::AbstractTensor{0}, y::Number)\n\nRaises the scalar-shaped tensor x to the power y. Returns a PowerOperation.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.sqrt-Tuple{AbstractTensor{0}}","page":"Operations","title":"Base.sqrt","text":"Base.sqrt(x::AbstractTensor{0})\n\nReturns the square root of x as a PowerOperation.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.:/-Tuple{AbstractTensor,AbstractTensor{0}}","page":"Operations","title":"Base.:/","text":"Base.:/(A::AbstractTensor, y::AbstractTensor{0})\n\nCreates a DivisionOperation where A is divived by y. y must be scalar-shaped.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.:/-Tuple{Number,AbstractTensor{0}}","page":"Operations","title":"Base.:/","text":"Base.:/(y::Number, A::AbstractTensor{0})\n\nCreates a DivisionOperation where y is divided by A, where A must be a scalar.\n\n\n\n\n\n","category":"method"},{"location":"operations/#TensorDSL.componenttensor-Tuple{AbstractTensor,Vararg{Index,N} where N}","page":"Operations","title":"TensorDSL.componenttensor","text":"componenttensor(A::AbstractTensor, indices::Vararg{Index})\n\nCreates a component tensor of A over indices. Indices must be a subset of the free indices of A.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.getindex-Tuple{AbstractTensor,Vararg{Index,N} where N}","page":"Operations","title":"Base.getindex","text":"Base.getindex(A::AbstractTensor, ys::Vararg{Index})\n\nCreates an IndexingOperation of A indexed by ys. If every dimension of A is not indexed, creates a ComponentTensorOperation over the unindexed dimensions.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.getindex-Tuple{AbstractTensor,Vararg{Union{Int64, String, Index},N} where N}","page":"Operations","title":"Base.getindex","text":"Base.getindex(A::AbstractTensor, ys::Vararg{Union{String, Int, Index}})\n\nA convenience function that allows indexing a tensor by an integer. The function creates a corresponding FixedIndex object in the appropiate vector space.\n\nTechnically the function also does the same for a string, but this is not recommended, because the syntax makes no distinction between upper and lower indices.\n\n\n\n\n\n","category":"method"},{"location":"operations/#Base.transpose-Tuple{AbstractTensor}","page":"Operations","title":"Base.transpose","text":"Base.transpose(A::AbstractTensor)\n\nReverses the shape of A. Does this by indexing A and then creating a componenttensor with the indices reversed.\n\n\n\n\n\n","category":"method"},{"location":"operations/#TensorDSL.trace-Tuple{AbstractTensor{2}}","page":"Operations","title":"TensorDSL.trace","text":"trace(A::AbstractTensor{2})\n\nReturns the trace of a matrix of the shape (V, V'). \n\n\n\n\n\n","category":"method"}]
}
