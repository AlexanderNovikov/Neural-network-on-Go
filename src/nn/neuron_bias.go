package nn

type BiasNeuron struct {
	*Neuron
}

func NewBiasNeuron() *BiasNeuron {
	n := new(BiasNeuron)
	n.Neuron = NewNeuron(NeuronTypeBias)
	n.value = 1.0
	return n
}
