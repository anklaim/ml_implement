package cbg

/*
#cgo linux LDFLAGS: -lcatboostmodel
#cgo darwin LDFLAGS: -lcatboostmodel
#include <stdlib.h>
#include <stdbool.h>
#include "../libs/model_interface/model_calcer_wrapper.h"

static char** makeCharArray(int size) {
    return calloc(sizeof(char*), size);
}

static void freeCharArray(char **a, int size) {
	int i;
	for (i = 0; i < size; i++)
		free(a[i]);
	free(a);
}

static char*** makeCharArray2(int size) {
	return calloc(sizeof(char**), size);
}

//тут надо сделать выделение памяти на одномерный массив под ону строку...кажется
static char** makeCharArray21d(int size) {
	return calloc(sizeof(char*), size);
}

static void freeCharArray2(char ***a, int sizeX, int sizeY) {
	int i;
	for (i = 0; i < sizeX; i++)
		freeCharArray(a[i], sizeY);
	free(a);
}

static float** makeFloatArray(int size) {
	return calloc(sizeof(float*), size);
}

static float* makeFloatArray1d(int size) {
    return calloc(sizeof(float), size);
}


static void setCharArray2(char ***a, char **s, int n){
	a[n] = s;
}

static void setFloatArray(float **a, float *f, int n){
	a[n] = f;
}

static void setFloatArray1d(float *a, float f, int n){
	a[n] = f;
}

static void setCharArray(char **a, char *s, int n) {
    a[n] = s;
}

*/
import "C"

import (
	"fmt"
	"unsafe"
)

func getError() error {
	messageC := C.GetErrorString()
	message := C.GoString(messageC)
	return fmt.Errorf(message)
}

// Model is a wrapper over ModelCalcerHandler
type Model struct {
	Handler unsafe.Pointer
}

// GetFloatFeaturesCount returns a number of float features used for training
func (model *Model) GetFloatFeaturesCount() int {
	return int(C.GetFloatFeaturesCount(model.Handler))
}

// GetCatFeaturesCount returns a number of categorical features used for training
func (model *Model) GetCatFeaturesCount() int {
	return int(C.GetCatFeaturesCount(model.Handler))
}

// Close deletes model handler
func (model *Model) Close() {
	C.ModelCalcerDelete(model.Handler)
}

// LoadFullModelFromFile loads model from file
func LoadFullModelFromFile(filename string) (*Model, error) {
	model := &Model{}
	model.Handler = C.ModelCalcerCreate()
	if !C.LoadFullModelFromFile(model.Handler, C.CString(filename)) {
		return nil, getError()
	}
	return model, nil
}

// CalcModelPrediction returns raw predictions for specified data points
func (model *Model) CalcModelPrediction(floats [][]float32, floatLength int, cats [][]string, catLength int) ([]float64, error) {

	nSamples := len(floats)
	results := make([]float64, nSamples)

	floatsC := C.makeFloatArray(C.int(nSamples))
	defer C.free(unsafe.Pointer(floatsC))
	for i, v := range floats {
		C.setFloatArray(floatsC, (*C.float)(&v[0]), C.int(i))
	}

	catsC := C.makeCharArray2(C.int(nSamples))
	defer C.freeCharArray2(catsC, C.int(nSamples), C.int(catLength))
	for i, cat := range cats {
		catC := C.makeCharArray(C.int(len(cat)))
		for i, c := range cat {
			C.setCharArray(catC, C.CString(c), C.int(i))
		}
		C.setCharArray2(catsC, catC, C.int(i))
	}

	if !C.CalcModelPrediction(
		model.Handler,
		C.size_t(nSamples),
		floatsC,
		C.size_t(floatLength),
		catsC,
		C.size_t(catLength),
		(*C.double)(&results[0]),
		C.size_t(nSamples),
	) {
		return nil, getError()
	}

	return results, nil
}

// CalcModelPredictionSingle returns a raw prediction for the specified data point
func (model *Model) CalcModelPredictionSingle(floats []float32, cats []string) (float64, error) {
	var result float64

	// Handling float features
	var floatsC *C.float
	if len(floats) > 0 {
		floatsC = C.makeFloatArray1d(C.int(len(floats)))
		defer C.free(unsafe.Pointer(floatsC))
		for i, f := range floats {
			*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(floatsC)) + uintptr(i)*unsafe.Sizeof(*floatsC))) = C.float(f)
		}
	} else {
		floatsC = nil // Если срез пуст, установите указатель на nil
	}

	// Handling categorical features
	catsC := C.makeCharArray(C.int(len(cats)))
	defer C.free(unsafe.Pointer(catsC))

	catStrings := make([]*C.char, len(cats)) // Дополнительный срез для хранения указателей на C-строки
	for i, cat := range cats {
		catStrings[i] = C.CString(cat)
		defer C.free(unsafe.Pointer(catStrings[i])) // Освободить память для каждой C-строки
		C.setCharArray(catsC, catStrings[i], C.int(i))
	}

	// Making prediction using single sample
	if !C.CalcModelPredictionSingle(
		model.Handler,
		floatsC,
		C.size_t(len(floats)),
		catsC,
		C.size_t(len(cats)),
		(*C.double)(&result),
		C.size_t(1),
	) {
		return 0, getError()
	}

	return result, nil
}
