# Setup

## Oolong
In Julia-REPL, enter package-mode by typing "]". Then write:
```
add https://github.com/oolongtensors/oolong
```


## Compilation
To compile Oolong to GEM, one needs to first [install Firedrake](https://firedrakeproject.org/download.html).
Then they should activate the Firedrake virtual environment and run the following commands ([source](https://github.com/JuliaPy/PyCall.jl/issues/706#issuecomment-610316716)):
```
(firedrake) > julia
julia> ENV["PYTHON"] = Sys.which("python")
julia> using Pkg
julia> Pkg.build("PyCall")
```

These commands make the PyCall-Python point to Firedrake's Python.