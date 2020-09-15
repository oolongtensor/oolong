using Documenter, TensorDSL

makedocs(sitename="Oolong",
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = !("local" in ARGS),
    ),
    modules = [TensorDSL],
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Tensors" => "tensors.md",
            "Operations" => "operations.md",
        ],
    ]
)