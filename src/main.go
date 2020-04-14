package main

import (
	"flag"
	"fmt"
	"log"
	"nn-go/src/nn"
	"nn-go/src/utils"
	"strconv"
	"strings"
)

type hiddenNeurons []int

func (h *hiddenNeurons) String() string {
	return ""
}

func (h *hiddenNeurons) Set(value string) error {
	*h = nil
	for _, i := range strings.Split(value, ",") {
		parsed, _ := strconv.Atoi(i)
		*h = append(*h, parsed)
	}
	return nil
}

func main() {
	var appDir, stateFile string
	var inputNeurons, outputNeurons, epochs int
	var hiddenNeurons hiddenNeurons = []int{2}
	var learningRate, moment float64
	var withBias, restore, validate bool

	flag.StringVar(&appDir, "appDir", "/opt/nn", "App directory")
	flag.StringVar(&stateFile, "stateFile", "/state/state.json", "State file")
	flag.IntVar(&epochs, "epochs", 10, "Number of epochs")
	flag.IntVar(&inputNeurons, "inputNeurons", 2, "Number of input neurons")
	flag.Var(&hiddenNeurons, "hiddenNeurons", "Number of hidden neurons")
	flag.IntVar(&outputNeurons, "outputNeurons", 1, "Number of output neurons")
	learningRate = *flag.Float64("learningRate", 0.7, "Learning rate")
	moment = *flag.Float64("moment", 0.3, "Moment")
	flag.BoolVar(&withBias, "withBias", true, "With bias")
	flag.BoolVar(&restore, "restore", false, "State restore")
	flag.BoolVar(&validate, "validate", false, "Validate")
	flag.Parse()

	var n *nn.Network
	if restore {
		n = nn.NewNetworkRestore(learningRate, moment)
		n.Restore(fmt.Sprintf("%s%s", appDir, stateFile))
	} else {
		n = nn.NewNetwork(inputNeurons, hiddenNeurons, outputNeurons, learningRate, moment, withBias)
	}

	if validate {
		images, _ := utils.ReadImages(fmt.Sprintf("%s/sets/t10k-images-idx3-ubyte", appDir))
		labels, _ := utils.ReadLabels(fmt.Sprintf("%s/sets/t10k-labels-idx1-ubyte", appDir))

		var goods int = 0
		for imageIndex, image := range images {
			forwardPassResult, err := n.ForwardPassOne(image)
			if err != nil {
				fmt.Println(err)
				return
			}

			resultIndex, _ := findMax(forwardPassResult)
			labelIndex, _ := findMax(labels[imageIndex])
			if resultIndex == labelIndex {
				goods++
			}
		}

		log.Println(fmt.Sprintf("Accuracy: %d %%", goods*100.0/len(images)))
	} else {
		images, _ := utils.ReadImages(fmt.Sprintf("%s/sets/train-images-idx3-ubyte", appDir))
		labels, _ := utils.ReadLabels(fmt.Sprintf("%s/sets/train-labels-idx1-ubyte", appDir))

		log.Println("Started")
		for epoch := 0; epoch < epochs; epoch++ {
			images, labels = utils.Shuffle(images, labels)
			for imageIndex, image := range images {
				_, err := n.ForwardPassOne(image)
				if err != nil {
					fmt.Println(err)
					return
				}

				n.BackpropagateOne(labels[imageIndex])
			}

			n.Save(fmt.Sprintf("%s%s", appDir, stateFile))
			log.Println(fmt.Sprintf("Epoch %d of %d", epoch, epochs))
		}
	}
}

func findMax(r []float64) (int, float64) {
	var maxIndex int
	var max float64 = 0.0
	for iIndex, i := range r {
		if i > max {
			maxIndex = iIndex
			max = i
		}
	}
	return maxIndex, max
}
