package nn

type LayerType int

const (
	LayerTypeInput  LayerType = 0
	LayerTypeHidden LayerType = 1
	LayerTypeOutput LayerType = 2
)

type ILayer interface {
	GetLayerType() LayerType
	SetLayerType(LayerType)
	GetNumberOfNeurons() int
	SetNumberOfNeurons(int)
	GetNeurons() []INeuron
	SetNeurons([]INeuron)
	ForwardPass(activationFunction ActivationFunction)
	GetWeights() [][]float64
}

type Layer struct {
	layerType       LayerType
	numberOfNeurons int
	neurons         []INeuron
}

func NewLayer(layerType LayerType, numberOfNeurons int) *Layer {
	l := new(Layer)
	l.layerType = layerType
	l.numberOfNeurons = numberOfNeurons
	return l
}

func (l *Layer) GetLayerType() LayerType {
	return l.layerType
}

func (l *Layer) SetLayerType(value LayerType) {
	l.layerType = value
}

func (l *Layer) GetNumberOfNeurons() int {
	return l.numberOfNeurons
}

func (l *Layer) SetNumberOfNeurons(value int) {
	l.numberOfNeurons = value
}

func (l *Layer) GetNeurons() []INeuron {
	return l.neurons
}

func (l *Layer) SetNeurons(value []INeuron) {
	l.neurons = value
}

func (l *Layer) ForwardPass(activationFunction ActivationFunction) {
	for _, neuron := range l.GetNeurons() {
		calculatedValue := 0.0
		for _, synapse := range neuron.GetInSynapses() {
			weight := synapse.Weight
			value := synapse.InNeuron.GetValue()
			calculatedValue += weight * value
		}

		neuron.SetValue(activationFunction(calculatedValue))
	}
}

func (l *Layer) GetWeights() [][]float64 {
	var layerWeights [][]float64
	for _, neuron := range l.neurons {
		var neuronWeights []float64
		for _, synapse := range neuron.GetInSynapses() {
			if synapse.InNeuron.GetNeuronType() == NeuronTypeBias {
				continue
			}
			neuronWeights = append(neuronWeights, synapse.Weight)
		}
		layerWeights = append(layerWeights, neuronWeights)
	}
	return layerWeights
}
