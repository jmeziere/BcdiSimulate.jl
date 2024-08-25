module BcdiSimulate
    using CUDA
    using LAMMPS
    using Random
    using Distributions
    using BcdiCore

    include("AtomicSimulate.jl")
end
