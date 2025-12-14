"""
Tests for modeling module.
"""
import unittest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestFusionNN(unittest.TestCase):
    """Test FusionNN architecture."""
    
    def setUp(self):
        """Import FusionNN components."""
        from hypopredict.modeling.fusion import FusionNN, ConcatFusion, AttentionFusion
        self.FusionNN = FusionNN
        self.ConcatFusion = ConcatFusion
        self.AttentionFusion = AttentionFusion
    
    def test_model_creation(self):
        """Test creating a FusionNN model."""
        model = self.FusionNN(
            ecg_input_dim=1,
            hidden_dim=128,
            num_classes=2
        )
        self.assertIsInstance(model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = self.FusionNN(
            ecg_input_dim=1,
            hidden_dim=64,
            num_classes=2
        )
        
        # Create mock ECG data (batch=2, length=250, channels=1)
        ecg = torch.randn(2, 250, 1)
        
        # Forward pass
        logits = model(ecg)
        
        # Check output shape
        self.assertEqual(logits.shape, (2, 2))  # batch_size=2, num_classes=2
    
    def test_predict(self):
        """Test prediction method."""
        model = self.FusionNN(
            ecg_input_dim=1,
            hidden_dim=64,
            num_classes=2
        )
        model.eval()
        
        ecg = torch.randn(2, 250, 1)
        predictions = model.predict(ecg)
        
        # Check that predictions are class labels (0 or 1)
        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(torch.all((predictions >= 0) & (predictions < 2)))
    
    def test_predict_proba(self):
        """Test probability prediction method."""
        model = self.FusionNN(
            ecg_input_dim=1,
            hidden_dim=64,
            num_classes=2
        )
        model.eval()
        
        ecg = torch.randn(2, 250, 1)
        probabilities = model.predict_proba(ecg)
        
        # Check that probabilities sum to 1
        self.assertEqual(probabilities.shape, (2, 2))
        prob_sums = probabilities.sum(dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(2), atol=1e-5))
    
    def test_attention_fusion(self):
        """Test attention-based fusion."""
        model = self.FusionNN(
            ecg_input_dim=1,
            hidden_dim=64,
            num_classes=2,
            use_attention=True
        )
        self.assertIsInstance(model.fusion, self.AttentionFusion)
    
    def test_concat_fusion(self):
        """Test concatenation-based fusion."""
        model = self.FusionNN(
            ecg_input_dim=1,
            hidden_dim=64,
            num_classes=2,
            use_attention=False
        )
        self.assertIsInstance(model.fusion, self.ConcatFusion)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestFusionLayers(unittest.TestCase):
    """Test fusion layer implementations."""
    
    def setUp(self):
        """Import fusion components."""
        from hypopredict.modeling.fusion import ConcatFusion, AttentionFusion
        self.ConcatFusion = ConcatFusion
        self.AttentionFusion = AttentionFusion
    
    def test_concat_fusion_single(self):
        """Test ConcatFusion with single encoding."""
        fusion = self.ConcatFusion(hidden_dim=64)
        encoding = torch.randn(2, 64)
        
        result = fusion(encoding)
        self.assertEqual(result.shape, encoding.shape)
    
    def test_attention_fusion_single(self):
        """Test AttentionFusion with single encoding."""
        fusion = self.AttentionFusion(hidden_dim=64)
        encoding = torch.randn(2, 64)
        
        result = fusion(encoding)
        self.assertEqual(result.shape, encoding.shape)


if __name__ == '__main__':
    unittest.main()
