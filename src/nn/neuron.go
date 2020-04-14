package nn

type NeuronType int

const (
	NeuronTypeInput  NeuronType = 0
	NeuronTypeHidden NeuronType = 1
	NeuronTypeOutput NeuronType = 2
	NeuronTypeBias   NeuronType = 3
)

type INeuron interface {
	GetValue() float64
	SetValue(float64)
	GetNeuronType() NeuronType
	SetNeuronType(NeuronType)
	GetInSynapses() []*Synapse
	SetInSynapses([]*Synapse)
	GetOutSynapses() []*Synapse
	SetOutSynapses([]*Synapse)
	AddSynapse(*Synapse, SynapseType)
}

type Neuron struct {
	value       float64
	neuronType  NeuronType
	inSynapses  []*Synapse
	outSynapses []*Synapse
}

func NewNeuron(neuronType NeuronType) *Neuron {
	n := new(Neuron)
	n.neuronType = neuronType
	return n
}

func (s *Neuron) AddSynapse(synapse *Synapse, synapseType SynapseType) {
	switch synapseType {
	case SynapseTypeIn:
		s.inSynapses = append(s.inSynapses, synapse)
		break
	case SynapseTypeOut:
		s.outSynapses = append(s.outSynapses, synapse)
		break
	}
}

func (n *Neuron) GetValue() float64 {
	return n.value
}

func (n *Neuron) SetValue(value float64) {
	n.value = value
}

func (n *Neuron) GetNeuronType() NeuronType {
	return n.neuronType
}

func (n *Neuron) SetNeuronType(value NeuronType) {
	n.neuronType = value
}

func (n *Neuron) GetInSynapses() []*Synapse {
	return n.inSynapses
}

func (n *Neuron) SetInSynapses(value []*Synapse) {
	n.inSynapses = value
}

func (n *Neuron) GetOutSynapses() []*Synapse {
	return n.outSynapses
}

func (n *Neuron) SetOutSynapses(value []*Synapse) {
	n.outSynapses = value
}
