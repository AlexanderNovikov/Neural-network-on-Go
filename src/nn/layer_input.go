package nn

type InputLayer struct {
	*Layer
}

func NewInputLayer(numberOfNeurons int) *InputLayer {
	l := new(InputLayer)
	l.Layer = NewLayer(LayerTypeInput, numberOfNeurons)
	l.build()
	return l
}

func (inputLayer *InputLayer) build() {
	var neurons []INeuron
	for i := 0; i < inputLayer.numberOfNeurons; i++ {
		neurons = append(neurons, NewInputNeuron())
	}
	inputLayer.neurons = neurons
}

func (inputLayer *InputLayer) SetInputData(input []float64) {
	for k, inputValue := range input {
		inputLayer.neurons[k].(*InputNeuron).SetValue(inputValue)
	}
}
