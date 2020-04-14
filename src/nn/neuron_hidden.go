package nn

type HiddenNeuron struct {
	*Neuron
	delta float64
}

func NewHiddenNeuron() *HiddenNeuron {
	n := new(HiddenNeuron)
	n.Neuron = NewNeuron(NeuronTypeHidden)
	return n
}

func (n *HiddenNeuron) GetDelta() float64 {
	return n.delta
}

func (n *HiddenNeuron) UpdateDelta(activationDerivativeFunction ActivationDerivativeFunction) float64 {
	n.delta = activationDerivativeFunction(n.value)
	calculatedValue := 0.0
	for _, synapse := range n.outSynapses {
		weight := synapse.Weight
		outNeuron := synapse.OutNeuron
		outNeuronType := outNeuron.GetNeuronType()
		switch outNeuronType {
		case NeuronTypeHidden:
			calculatedValue += weight * outNeuron.(*HiddenNeuron).GetDelta()
			break
		case NeuronTypeOutput:
			calculatedValue += weight * outNeuron.(*OutputNeuron).GetDelta()
			break
		}
	}
	n.delta *= calculatedValue
	return n.delta
}
