package nn

type HiddenLayer struct {
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

func NewHiddenLayer(numberOfNeurons int, prevLayer ILayer, withBias bool, weights [][]float64, biasWeights []float64, activationDerivativeFunction ActivationDerivativeFunction, learningRate, moment float64) *HiddenLayer {
	l := new(HiddenLayer)
	l.Layer = NewLayer(LayerTypeHidden, numberOfNeurons)
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

func (hiddenLayer *HiddenLayer) build() {
	var neurons []INeuron
	biasNeuron := NewBiasNeuron()
	for i := 0; i < hiddenLayer.numberOfNeurons; i++ {
		hiddenNeuron := NewHiddenNeuron()
		var neuronWeights []float64
		if hiddenLayer.weights != nil && i < len(hiddenLayer.weights) {
			neuronWeights = hiddenLayer.weights[i]
		}
		for prevLayerNeuronIndex, prevLayerNeuron := range hiddenLayer.prevLayer.GetNeurons() {
			synapse := NewSynapse(hiddenLayer.learningRate, hiddenLayer.moment)
			if neuronWeights != nil && prevLayerNeuronIndex < len(neuronWeights) {
				synapse.Weight = neuronWeights[prevLayerNeuronIndex]
			}
			hiddenNeuron.AddSynapse(synapse, SynapseTypeIn)
			prevLayerNeuron.AddSynapse(synapse, SynapseTypeOut)
			synapse.InNeuron = prevLayerNeuron
			synapse.OutNeuron = hiddenNeuron
		}

		if hiddenLayer.withBias {
			synapse := NewSynapse(hiddenLayer.learningRate, hiddenLayer.moment)
			if hiddenLayer.biasWeights != nil && i < len(hiddenLayer.biasWeights) {
				synapse.Weight = hiddenLayer.biasWeights[i]
			}
			hiddenNeuron.AddSynapse(synapse, SynapseTypeIn)
			biasNeuron.AddSynapse(synapse, SynapseTypeOut)
			synapse.InNeuron = biasNeuron
			synapse.OutNeuron = hiddenNeuron
		}

		neurons = append(neurons, hiddenNeuron)
	}
	if hiddenLayer.withBias {
		hiddenLayer.biasNeuron = biasNeuron
	}
	hiddenLayer.neurons = neurons
}

func (hiddenLayer *HiddenLayer) UpdateDelta() {
	for _, neuron := range hiddenLayer.neurons {
		neuron.(*HiddenNeuron).UpdateDelta(hiddenLayer.activationDerivativeFunction)
	}
}

func (hiddenLayer *HiddenLayer) UpdateWeight() {
	for _, neuron := range hiddenLayer.neurons {
		for _, synapse := range neuron.GetInSynapses() {
			if synapse.InNeuron.GetNeuronType() == NeuronTypeBias {
				continue
			}
			synapse.UpdateGradient()
			synapse.UpdateWeight()
		}
	}
}

func (hiddenLayer *HiddenLayer) GetBiasWeights() []float64 {
	var layerBiasWeights []float64
	if hiddenLayer.withBias {
		for _, synapse := range hiddenLayer.biasNeuron.outSynapses {
			layerBiasWeights = append(layerBiasWeights, synapse.Weight)
		}
	}
	return layerBiasWeights
}
