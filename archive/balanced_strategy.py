# Balanced Strategy Configuration
# Generated with optimized thresholds


class BalancedForexStrategy:
    """Strategy with advanced balanced signal processing"""
    
    def __init__(self, model_path="data/models/optimized_forex_ai.pth"):
        self.model = SimpleOptimizedLSTM(input_size=20, hidden_size=128, num_layers=3, dropout=0.4)
        self.load_model(model_path)
        
        # Advanced balanced thresholds
        self.thresholds = {
            "buy_threshold": 0.452,
            "sell_threshold": 0.350,
            "hold_preference": 0.100,
            "confidence_boost": 1.2
        }
    
    def get_balanced_signal(self, dataframe):
        """Generate balanced trading signal"""
        features = create_simple_features(dataframe, target_features=20)
        
        if len(features) >= 80:
            sequence = features[-80:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(sequence_tensor)
                signal, confidence = self.process_balanced_signal(output)
                
                return {
                    "action": signal,
                    "confidence": confidence,
                    "balanced": True,
                    "processing_time": "~0.4s"
                }
        
        return {"action": "HOLD", "confidence": 0.5, "error": "Insufficient data"}
