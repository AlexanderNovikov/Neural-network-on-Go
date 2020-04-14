package nn

type InputNeuron struct {
	*Neuron
}

func NewInputNeuron() *InputNeuron {
	n := new(InputNeuron)
	n.Neuron = NewNeuron(NeuronTypeInput)
	return n
}

func (n *InputNeuron) SetValue(value float64) {
	if value > 1 {
		value = 1 / value
	}
	n.value = value
}
