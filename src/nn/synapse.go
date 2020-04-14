package nn

import "math/rand"

type SynapseType int

const (
	SynapseTypeIn  SynapseType = 0
	SynapseTypeOut SynapseType = 1
)

type Synapse struct {
	Weight       float64
	WeightDelta  float64
	InNeuron     INeuron
	OutNeuron    INeuron
	Gradient     float64
	learningRate float64
	moment       float64
}

func NewSynapse(learningRate, moment float64) *Synapse {
	s := new(Synapse)
	s.Weight = -1 + rand.Float64()*(1 - -1)
	s.learningRate = learningRate
	s.moment = moment
	return s
}

func (s *Synapse) UpdateGradient() float64 {
	s.Gradient = s.InNeuron.GetValue()
	outNeuron := s.OutNeuron
	outNeuronType := outNeuron.GetNeuronType()
	switch outNeuronType {
	case NeuronTypeHidden:
		s.Gradient *= outNeuron.(*HiddenNeuron).GetDelta()
		break
	case NeuronTypeOutput:
		s.Gradient *= outNeuron.(*OutputNeuron).GetDelta()
		break
	}
	return s.Gradient
}

func (s *Synapse) UpdateWeight() float64 {
	s.WeightDelta = s.learningRate * s.Gradient
	s.Weight += s.WeightDelta
	return s.Weight
}
