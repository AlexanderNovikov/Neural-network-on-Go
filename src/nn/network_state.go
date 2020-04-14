package nn

type NetworkState struct {
	NumberOfInputNeurons  int
	NumberOfHiddenNeurons []int
	NumberOfOutputNeurons int
	WithBias              bool
	HiddenWeights         [][][]float64
	HiddenBiasWeights     [][]float64
	OutputWeights         [][]float64
	OutputBiasWeights     []float64
}
