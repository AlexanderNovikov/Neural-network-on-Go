package utils

import (
	"math/rand"
)

func Shuffle(a, b [][]float64) ([][]float64, [][]float64) {
	for i := len(a) - 1; i > 0; i-- {
		rVal := rand.Intn(i)
		tempA := a[i]
		tempB := b[i]
		a[i] = a[rVal]
		b[i] = b[rVal]
		a[rVal] = tempA
		b[rVal] = tempB
	}
	return a, b
}
