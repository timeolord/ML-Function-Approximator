using Flux, Plots, StatsBase

function generateData(data=100)
    x1 = Float32[]
    y1 = Float32[]
    for steps in 0:data
        value = steps/20
        push!(x1, value)
        push!(y1, tan(value))
    end
    return standardize(UnitRangeTransform, x1),y1
end

function train(epochs, model, data)
    Flux.@epochs epochs Flux.train!(Loss, params(model), data, Descent())
end

datax, datay = generateData(100) 
data = zip(datax, datay) 

model = Chain(Dense(1,25,relu),
        Dense(25,25,relu),Dense(25,25,relu),
        Dense(25,1,tanh)) 

Loss(x,y) = Flux.mse(model([x]), y) 

train(10000, model, data)

#Below is all for plotting

y2 = Float32[]

for x in datax
    push!(y2, model([x])[1])
end


plot(datax,hcat(y2,datay))
println("Did it go better this time?")

#println(sin(pi/8))
#println(model([pi/8]))
#Loss(pi/8,sin(pi/8))




