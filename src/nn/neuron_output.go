package nn

type OutputNeuron struct {
	*Neuron
	delta float64
}

func NewOutputNeuron() *OutputNeuron {
	n := new(OutputNeuron)
	n.Neuron = NewNeuron(NeuronTypeOutput)
	return n
}

func (n *OutputNeuron) GetDelta() float64 {
	return n.delta
}

func (n *OutputNeuron) UpdateDelta(input float64, activationDerivativeFunction ActivationDerivativeFunction) float64 {
	n.delta = (input - n.GetValue()) * activationDerivativeFunction(n.GetValue())
	return n.delta
}
