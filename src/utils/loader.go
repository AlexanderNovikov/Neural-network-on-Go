package utils

import (
	"math"
	"os"
)

const (
	imagesFileMagic = 0x00000803
	labelsFileMagic = 0x00000801
)

var (
	labelTemplate = []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
)

func ReadImages(path string) ([][]float64, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	magic, e := readInt32(f)
	if e != nil || magic != imagesFileMagic {
		return nil, e
	}
	n, e := readInt32(f)
	if e != nil {
		return nil, e
	}
	w, e := readInt32(f)
	if e != nil {
		return nil, e
	}
	h, e := readInt32(f)
	if e != nil {
		return nil, e
	}
	sz := n * w * h
	data := make([]uint8, sz)
	length, e := f.Read(data)
	if e != nil || length != sz {
		return nil, e
	}

	var newData [][]float64
	for i := 0; i < len(data); i += w * h {
		var image []float64
		for _, b := range data[i:int(math.Min(float64(len(data)), float64(i+w*h)))] {
			image = append(image, float64(b))
		}
		newData = append(newData, image)
	}
	return newData[0:int(math.Min(float64(len(newData)), 50000))], nil
}

func ReadLabels(path string) ([][]float64, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	magic, e := readInt32(f)
	if e != nil || magic != labelsFileMagic {
		return nil, e
	}
	n, e := readInt32(f)
	if e != nil {
		return nil, e
	}
	data := make([]uint8, n)
	length, e := f.Read(data)
	if e != nil || length != n {
		return nil, e
	}
	var newData [][]float64
	for _, b := range data {
		var label []float64
		label = append(label, labelTemplate...)
		label[int(b)] = 1.0
		newData = append(newData, label)
	}
	return newData[0:int(math.Min(float64(len(newData)), 50000))], nil
}

func readInt32(f *os.File) (int, error) {
	buf := make([]byte, 4)
	n, e := f.Read(buf)
	switch {
	case e != nil:
		return 0, e
	case n != 4:
		return 0, e
	}
	v := 0
	for _, x := range buf {
		v = v*256 + int(x)
	}
	return v, nil
}
