using Flux, Plots

function generateData()
  xValues = Float32[]
  yValues = Float32[]
  for step in 0:1000
    x = step/10
    push!(xValues, x)
    push!(yValues, sin(x))
  end
  return xValues, yValues
end

xValues, yValues = generateData()

model = Chain(Dense(5, 5, sigmoid), Dense(5, 1, sigmoid))

function getInput(i)
  return [xValues[i], xValues[i+1], xValues[i+2], xValues[i+3], xValues[i+4]]
end

function training(model)
  local training_loss
  parameters = params(model)
  for i in 1:(lastindex(xValues)-5)
    grads = gradient(parameters) do 
      training_loss = Flux.Losses.mse(yValues[i], model(getInput(i)))
      #@show training_loss
      return training_loss
    end
    Flux.update!(Flux.Optimise.Descent(), parameters, grads)
   
  end
end

Flux.@epochs 150 training(model)

training_loss = Flux.Losses.mse(yValues[5], model(getInput(5)))

@show training_loss

function predictValues(list)
  values = Float32[]
  for i in 1:(lastindex(list)-5)
    push!(values, model(getInput(i))[1])
  end
  for j in 1:5
    push!(values, 0)
  end
  return values
end

predictedValues = predictValues(xValues)

plot([yValues predictedValues])