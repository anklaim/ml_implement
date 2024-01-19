package ml_implement

import (
	cbg "github.com/anklaim/ml_implement/cbg"
)

// на вход подается массив строк со значениями {HOLD, BUY, MISSING}
// размера 40
// очередности
//
//	"AO_15m",
//	"AO_1h",
//	"AO_1m",
//	"AO_5m",
//	"AROON_15m",
//	"AROON_1h",
//	"AROON_1m",
//	"AROON_5m",
//	"BB_15m",
//	"BB_1h",
//	"BB_1m",
//	"BB_5m",
//	"HAMMER_15m",
//	"HAMMER_1h",
//	"HAMMER_1m",
//	"HAMMER_5m",
//	"MACD_15m",
//	"MACD_1h",
//	"MACD_1m",
//	"MACD_5m",
//	"RSI_15m",
//	"RSI_1h",
//	"RSI_1m",
//	"RSI_5m",
//	"SCALPING_SCALPING",
//	"SCALP_15M_15m",
//	"SCALP_1H_1h",
//	"SCALP_1M_1m",
//	"SCALP_5M_5m",
//	"SMA_15m",
//	"SMA_1h",
//	"SMA_1m",
//	"SMA_5m",
//	"SO_15m",
//	"SO_1h",
//	"SO_1m",
//	"SO_5m",
//	"BREAKOUT_BREAKOUT",
//	"ORDER_FLOW_ORDER_FLOW",
//	"INSTANT_PRICE_INSTANT_PRICE",
func get_catboost_predict(categories []string) float64 {
	model, _ := cbg.LoadBinaryClassifierFromFile("cbm_catboost_model.cbm")
	// 7      fmt.Println(err)
	numbers := make([]float32, 0)
	prob, _ := model.PredictProba(numbers, categories)
	return prob
}

func get_model(filename string) (*cbg.BinaryClassifer, error) {
	return cbg.LoadBinaryClassifierFromFile("cbm_catboost_model.cbm")
}
