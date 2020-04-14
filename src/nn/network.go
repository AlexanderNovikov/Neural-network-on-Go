package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"time"
)

type ActivationFunction func(value float64) float64
type ActivationDerivativeFunction func(value float64) float64

type Network struct {
	NumberOfInputNeurons         int
	NumberOfHiddenNeurons        []int
	NumberOfOutputNeurons        int
	InputLayer                   *InputLayer
	HiddenLayers                 []*HiddenLayer
	OutputLayer                  *OutputLayer
	Layers                       []ILayer
	learningRate                 float64
	moment                       float64
	withBias                     bool
	activationFunction           ActivationFunction
	activationDerivativeFunction ActivationDerivativeFunction
	hiddenWeights                [][][]float64
	hiddenBiasWeights            [][]float64
	outputWeights                [][]float64
	outputBiasWeights            []float64
}

func NewNetwork(numberOfInputNeurons int, numberOfHiddenNeurons []int, numberOfOutputNeurons int, learningRate, moment float64, withBias bool) *Network {
	rand.Seed(time.Now().UnixNano())
	n := new(Network)
	n.NumberOfInputNeurons = numberOfInputNeurons
	n.NumberOfHiddenNeurons = numberOfHiddenNeurons
	n.NumberOfOutputNeurons = numberOfOutputNeurons
	n.learningRate = learningRate
	n.moment = moment
	n.withBias = withBias
	if true {
		n.activationFunction = func(value float64) float64 {
			return 1 / (1 + math.Exp(-value))
		}
		n.activationDerivativeFunction = func(value float64) float64 {
			return (1 - value) * value
		}
	}
	n.build()
	return n
}

func NewNetworkRestore(learningRate, moment float64) *Network {
	n := new(Network)
	n.learningRate = learningRate
	n.moment = moment
	if true {
		n.activationFunction = func(value float64) float64 {
			return 1 / (1 + math.Exp(-value))
		}
		n.activationDerivativeFunction = func(value float64) float64 {
			return (1 - value) * value
		}
	}
	return n
}

func (n *Network) build() {
	var prevLayer ILayer
	n.InputLayer = NewInputLayer(n.NumberOfInputNeurons)
	n.Layers = append(n.Layers, n.InputLayer)
	prevLayer = n.InputLayer
	var hiddenLayers []*HiddenLayer
	for numberOfHiddenNeuronsIndex, numberOfHiddenNeurons := range n.NumberOfHiddenNeurons {
		var hiddenWeights [][]float64
		var hiddenBiasWeights []float64
		if n.hiddenWeights != nil && numberOfHiddenNeuronsIndex < len(n.hiddenWeights) {
			hiddenWeights = n.hiddenWeights[numberOfHiddenNeuronsIndex]
		}
		if n.withBias && n.hiddenBiasWeights != nil && numberOfHiddenNeuronsIndex < len(n.hiddenBiasWeights) {
			hiddenBiasWeights = n.hiddenBiasWeights[numberOfHiddenNeuronsIndex]
		}
		hiddenLayer := NewHiddenLayer(numberOfHiddenNeurons, prevLayer, n.withBias, hiddenWeights, hiddenBiasWeights, n.activationDerivativeFunction, n.learningRate, n.moment)
		hiddenLayers = append(hiddenLayers, hiddenLayer)
		n.Layers = append(n.Layers, hiddenLayer)
		prevLayer = hiddenLayer
	}
	n.HiddenLayers = hiddenLayers
	n.OutputLayer = NewOutputLayer(n.NumberOfOutputNeurons, prevLayer, n.withBias, n.outputWeights, n.outputBiasWeights, n.activationDerivativeFunction, n.learningRate, n.moment)
	n.Layers = append(n.Layers, n.OutputLayer)
}

func (n *Network) ForwardPassOne(input []float64) ([]float64, error) {
	if len(input) != n.NumberOfInputNeurons {
		return nil, fmt.Errorf("number of input values and input neurons does not match")
	}
	n.InputLayer.SetInputData(input)
	for i := 1; i < len(n.Layers); i++ {
		n.Layers[i].ForwardPass(n.activationFunction)
	}
	return n.OutputLayer.GetOutput(), nil
}

func (n *Network) UpdateDelta(input []float64) error {
	if len(input) != n.NumberOfOutputNeurons {
		return fmt.Errorf("number of ideal values and output neurons does not match")
	}
	n.OutputLayer.UpdateDelta(input)
	for _, layer := range n.HiddenLayers {
		layer.UpdateDelta()
	}
	return nil
}

func (n *Network) UpdateWeight() {
	n.OutputLayer.UpdateWeight()
	for _, layer := range n.HiddenLayers {
		layer.UpdateWeight()
	}
}

func (n *Network) BackpropagateOne(input []float64) error {
	if len(input) != n.NumberOfOutputNeurons {
		return fmt.Errorf("number of ideal values and output neurons does not match")
	}
	n.OutputLayer.UpdateDelta(input)
	n.OutputLayer.UpdateWeight()
	for _, layer := range n.HiddenLayers {
		layer.UpdateDelta()
		layer.UpdateWeight()
	}

	return nil
}

func (n *Network) GetError(input []float64) (float64, error) {
	if len(input) != n.NumberOfOutputNeurons {
		return 0, fmt.Errorf("number of ideal values and output neurons does not match")
	}
	return n.OutputLayer.GetError(input), nil
}

func (n *Network) Save(path string) {
	networkState := new(NetworkState)
	networkState.NumberOfInputNeurons = n.NumberOfInputNeurons
	networkState.NumberOfHiddenNeurons = n.NumberOfHiddenNeurons
	networkState.NumberOfOutputNeurons = n.NumberOfOutputNeurons
	networkState.WithBias = n.withBias
	var hiddenWeights [][][]float64
	var hiddenBiasWeights [][]float64
	for _, hiddenLayer := range n.HiddenLayers {
		hiddenWeights = append(hiddenWeights, hiddenLayer.GetWeights())
		hiddenBiasWeights = append(hiddenBiasWeights, hiddenLayer.GetBiasWeights())
	}
	networkState.HiddenWeights = hiddenWeights
	networkState.HiddenBiasWeights = hiddenBiasWeights
	networkState.OutputWeights = n.OutputLayer.GetWeights()
	networkState.OutputBiasWeights = n.OutputLayer.GetBiasWeights()
	b, err := json.Marshal(networkState)
	if err != nil {
		fmt.Println(err)
		return
	}
	err = ioutil.WriteFile(path, b, 0644)
	if err != nil {
		fmt.Println(err)
		return
	}
}

func (n *Network) Restore(path string) {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	networkState := new(NetworkState)
	err = json.Unmarshal(b, networkState)
	if err != nil {
		fmt.Println(err)
		return
	}
	n.NumberOfInputNeurons = networkState.NumberOfInputNeurons
	n.NumberOfHiddenNeurons = networkState.NumberOfHiddenNeurons
	n.NumberOfOutputNeurons = networkState.NumberOfOutputNeurons
	n.withBias = networkState.WithBias
	n.hiddenWeights = networkState.HiddenWeights
	n.hiddenBiasWeights = networkState.HiddenBiasWeights
	n.outputWeights = networkState.OutputWeights
	n.outputBiasWeights = networkState.OutputBiasWeights
	n.build()
}
