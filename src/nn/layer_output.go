package nn

import (
	"math"
)

type OutputLayer struct {
	*Layer
	biasNeuron                   *BiasNeuron
	prevLayer                    ILayer
	withBias                     bool
	weights                      [][]float64
	biasWeights                  []float64
	activationDerivativeFunction ActivationDerivativeFunction
	learningRate                 float64
	moment                       float64
}

func NewOutputLayer(numberOfNeurons int, prevLayer ILayer, withBias bool, weights [][]float64, biasWeights []float64, activationDerivativeFunction ActivationDerivativeFunction, learningRate, moment float64) *OutputLayer {
	l := new(OutputLayer)
	l.Layer = NewLayer(LayerTypeOutput, numberOfNeurons)
	l.prevLayer = prevLayer
	l.withBias = withBias
	l.weights = weights
	l.biasWeights = biasWeights
	l.activationDerivativeFunction = activationDerivativeFunction
	l.learningRate = learningRate
	l.moment = moment
	l.build()
	return l
}

func (outputLayer *OutputLayer) build() {
	var neurons []INeuron
	biasNeuron := NewBiasNeuron()
	for i := 0; i < outputLayer.numberOfNeurons; i++ {
		outputNeuron := NewOutputNeuron()
		var neuronWeights []float64
		if outputLayer.weights != nil && i < len(outputLayer.weights) {
			neuronWeights = outputLayer.weights[i]
		}
		for prevLayerNeuronIndex, prevLayerNeuron := range outputLayer.prevLayer.GetNeurons() {
			synapse := NewSynapse(outputLayer.learningRate, outputLayer.moment)
			if neuronWeights != nil && prevLayerNeuronIndex < len(neuronWeights) {
				synapse.Weight = neuronWeights[prevLayerNeuronIndex]
			}
			outputNeuron.AddSynapse(synapse, SynapseTypeIn)
			prevLayerNeuron.AddSynapse(synapse, SynapseTypeOut)
			synapse.InNeuron = prevLayerNeuron
			synapse.OutNeuron = outputNeuron
		}

		if outputLayer.withBias {
			synapse := NewSynapse(outputLayer.learningRate, outputLayer.moment)
			if outputLayer.biasWeights != nil && i < len(outputLayer.biasWeights) {
				synapse.Weight = outputLayer.biasWeights[i]
			}
			outputNeuron.AddSynapse(synapse, SynapseTypeIn)
			biasNeuron.AddSynapse(synapse, SynapseTypeOut)
			synapse.InNeuron = biasNeuron
			synapse.OutNeuron = outputNeuron
		}

		neurons = append(neurons, outputNeuron)
	}
	if outputLayer.withBias {
		outputLayer.biasNeuron = biasNeuron
	}
	outputLayer.neurons = neurons
}

func (outputLayer *OutputLayer) GetOutput() []float64 {
	var result []float64
	for _, outputNeuron := range outputLayer.neurons {
		result = append(result, outputNeuron.GetValue())
	}
	return result
}

func (outputLayer *OutputLayer) GetError(input []float64) float64 {
	calculatedValue := 0.0
	for neuronIndex, neuron := range outputLayer.neurons {
		idealInput := input[neuronIndex]
		calculatedValue += math.Pow(idealInput-neuron.GetValue(), 2)
	}
	calculatedValue /= float64(outputLayer.numberOfNeurons)

	return calculatedValue
}

func (outputLayer *OutputLayer) UpdateDelta(input []float64) {
	for neuronIndex, neuron := range outputLayer.neurons {
		inputValue := input[neuronIndex]
		neuron.(*OutputNeuron).UpdateDelta(inputValue, outputLayer.activationDerivativeFunction)
	}
}

func (outputLayer *OutputLayer) UpdateWeight() {
	for _, neuron := range outputLayer.neurons {
		for _, synapse := range neuron.GetInSynapses() {
			if synapse.InNeuron.GetNeuronType() == NeuronTypeBias {
				continue
			}
			synapse.UpdateGradient()
			synapse.UpdateWeight()
		}
	}
}

func (outputLayer *OutputLayer) GetBiasWeights() []float64 {
	var layerBiasWeights []float64
	if outputLayer.withBias {
		for _, synapse := range outputLayer.biasNeuron.outSynapses {
			layerBiasWeights = append(layerBiasWeights, synapse.Weight)
		}
	}
	return layerBiasWeights
}
