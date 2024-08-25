using BcdiSimulate
using Test
using Random
using Distributions

@testset "BcdiSimulate.jl" begin
    n = 20

    x = 100 .* rand(n)
    y = 100 .* rand(n)
    z = 100 .* rand(n)

    hStart = 20 * (rand() - 0.5)
    hRange = (hStart:0.005:hStart+0.02)
    kStart = 20 * (rand() - 0.5)
    kRange = (kStart:0.005:kStart+0.02)
    lStart = 20 * (rand() - 0.5)
    lRange = (lStart:0.005:lStart+0.02)

    numPhotons = 1e6

    Random.seed!(1)
    elecFieldTester = zeros(ComplexF64, length(hRange), length(kRange), length(lRange))
    for i in 1:length(hRange), j in 1:length(kRange), k in 1:length(lRange), l in 1:length(x)
        elecFieldTester[i,j,k] += exp(-1im * 2 * pi * (x[l] * hRange[i] +  y[l] * kRange[j] + z[l] * lRange[k]))
    end
    c = numPhotons / mapreduce(abs2, +, elecFieldTester)
    intensTester = rand.(Poisson.(c .* abs2.(elecFieldTester)))

    tmp, _, _, _ = BcdiSimulate.simulateElectricField(x, y, z, [hRange], [kRange], [lRange])
    elecFieldTestee = tmp[1]
    tmp, _, _, _ = BcdiSimulate.simulateDiffraction(x, y, z, [hRange], [kRange], [lRange], [numPhotons], seed=1)
    intensTestee = tmp[1]

    @test all(isapprox.(elecFieldTester, elecFieldTestee, rtol=1e-6))
    @test all(isapprox.(intensTester, intensTestee, rtol=1e-6))
end
