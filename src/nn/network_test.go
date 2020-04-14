package nn

import (
	"fmt"
	"os"
	"testing"
)

func TestCalcOne(t *testing.T) {
	network := NewNetworkRestore(0.7, 0.3)
	currentDir, _ := os.Getwd()
	network.Restore(fmt.Sprintf("%s/../../state/test-state.json", currentDir))
	result, err := network.ForwardPassOne([]float64{1.0, 0.0})
	if err != nil {
		fmt.Println(err)
	}
	if result[0] != 0.34049134000389103 {
		t.Error("Expected 0.34049134000389103 got ", result[0])
	}
}

func TestGetError(t *testing.T) {
	network := NewNetworkRestore(0.7, 0.3)
	currentDir, _ := os.Getwd()
	network.Restore(fmt.Sprintf("%s/../../state/test-state.json", currentDir))
	network.ForwardPassOne([]float64{1.0, 0.0})
	forwardPassError, err := network.GetError([]float64{1.0})
	if err != nil {
		fmt.Println(err)
	}

	if forwardPassError != 0.4349516726098633 {
		t.Error("Expected 0.4349516726098633 got ", forwardPassError)
	}
}

func TestBackpropagateOne(t *testing.T) {
	network := NewNetworkRestore(0.7, 0.3)
	currentDir, _ := os.Getwd()
	network.Restore(fmt.Sprintf("%s/../../state/test-state.json", currentDir))
	input := []float64{1.0, 0.0}
	ideal := []float64{1.0}
	resultBefore, _ := network.ForwardPassOne(input)
	forwardPassErrorBefore, _ := network.GetError(ideal)
	network.BackpropagateOne(ideal)
	resultAfter, _ := network.ForwardPassOne(input)
	forwardPassErrorAfter, _ := network.GetError(ideal)

	if resultBefore[0] != 0.34049134000389103 {
		t.Error("Expected 0.34049134000389103 got ", resultBefore[0])
	}
	if forwardPassErrorBefore != 0.4349516726098633 {
		t.Error("Expected 0.4349516726098633 got ", forwardPassErrorBefore)
	}
	if resultAfter[0] != 0.36927965478491614 {
		t.Error("Expected 0.36927965478491614 got ", resultAfter[0])
	}
	if forwardPassErrorAfter != 0.3978081538682345 {
		t.Error("Expected 0.3978081538682345 got ", forwardPassErrorAfter)
	}
}
