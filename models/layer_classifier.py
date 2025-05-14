"""
Neural network models for layer classification.
"""

import torch
import torch.nn as nn

class SupervisedChannelAutoencoder(nn.Module):
    """
    Supervised channel autoencoder for neural data processing.
    
    This model combines an autoencoder for learning a meaningful latent representation
    with a classifier for predicting cortical layers.
    """
    def __init__(self, 
                 model_name="channel_ae",
                 input_channels=3, 
                 latent_dim=64, 
                 maxC=128, 
                 num_classes=8):
        """
        Initialize the supervised channel autoencoder.
        
        Args:
            model_name (str): Name of the model (e.g., "PhaseMagnitude")
            input_channels (int): Number of input channels (3 for phase magnitude/angle, 75 for comod)
            latent_dim (int): Dimension of the latent space
            maxC (int): Maximum channel dimension
            num_classes (int): Number of cortical layer classes
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.maxC = maxC

        # Encoder CNN
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU()
        )

        # Compute flat_dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 1, maxC)
            out = self.encoder_cnn(dummy)
            self.flat_dim = out.view(1, -1).shape[1]

        self.encoder_fc = nn.Linear(self.flat_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, self.flat_dim)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(1,3), stride=(1,2), 
                               padding=(0,1), output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(1,3), stride=(1,2), 
                               padding=(0,1), output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, kernel_size=(1,3), stride=(1,2), 
                               padding=(0,1), output_padding=(0,1)),
            nn.Tanh()
        )

        # Classification head
        self.classifier = nn.Linear(latent_dim, num_classes)
    
    def forward(self, x, mask=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, input_channels, 1, maxC)
            mask (torch.Tensor, optional): Mask tensor to apply to input
            
        Returns:
            tuple: (recon, z, logits) - reconstructed input, latent features, and class logits
        """
        # Apply mask if provided
        if mask is not None:
            x = x * mask
            
        # Encoder
        out = self.encoder_cnn(x)                   # (B, 64, 1, ?)
        out = out.view(x.size(0), -1)
        z = self.encoder_fc(out)
        
        # Decoder
        out = self.decoder_fc(z).view(x.size(0), 64, 1, -1)
        recon = self.decoder_cnn(out)
        
        # Classifier
        logits = self.classifier(z)
        
        return recon, z, logits
    
    def get_latent_features(self, x, mask=None):
        """
        Extract latent features from the model.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor, optional): Mask tensor
            
        Returns:
            torch.Tensor: Latent feature vector
        """
        # Apply mask if provided
        if mask is not None:
            x = x * mask
            
        # Forward pass through encoder only
        out = self.encoder_cnn(x)
        out = out.view(x.size(0), -1)
        z = self.encoder_fc(out)
        
        return z


class PhaseMagnitudeLayerClassifier(SupervisedChannelAutoencoder):
    """
    Phase magnitude layer classifier model.
    """
    def __init__(self, num_classes=8, latent_dim=64, maxC=128):
        """
        Initialize the phase magnitude layer classifier.
        
        Args:
            num_classes (int): Number of output classes (cortical layers)
            latent_dim (int): Dimension of the latent space
            maxC (int): Maximum channel dimension
        """
        super(PhaseMagnitudeLayerClassifier, self).__init__(
            model_name="Phase Magnitude",
            input_channels=3,
            latent_dim=latent_dim,
            maxC=maxC,
            num_classes=num_classes
        )


class PhaseAngleLayerClassifier(SupervisedChannelAutoencoder):
    """
    Phase angle layer classifier model.
    """
    def __init__(self, num_classes=8, latent_dim=64, maxC=128):
        """
        Initialize the phase angle layer classifier.
        
        Args:
            num_classes (int): Number of output classes (cortical layers)
            latent_dim (int): Dimension of the latent space
            maxC (int): Maximum channel dimension
        """
        super(PhaseAngleLayerClassifier, self).__init__(
            model_name="Phase Angle",
            input_channels=3,
            latent_dim=latent_dim,
            maxC=maxC,
            num_classes=num_classes
        )


class ComodLayerClassifier(SupervisedChannelAutoencoder):
    """
    Comodulation layer classifier model.
    """
    def __init__(self, num_classes=8, latent_dim=64, maxC=128):
        """
        Initialize the comodulation layer classifier.
        
        Args:
            num_classes (int): Number of output classes (cortical layers)
            latent_dim (int): Dimension of the latent space
            maxC (int): Maximum channel dimension
        """
        super(ComodLayerClassifier, self).__init__(
            model_name="Comodulation",
            input_channels=75,
            latent_dim=latent_dim,
            maxC=maxC,
            num_classes=num_classes
        )


class EnsembleClassifier(nn.Module):
    """
    Ensemble model that combines predictions from multiple classifiers.
    """
    def __init__(self, num_classes=8):
        """
        Initialize the ensemble classifier.
        
        Args:
            num_classes (int): Number of output classes (cortical layers)
        """
        super().__init__()
        
        # Individual models
        self.phase_mag_model = PhaseMagnitudeLayerClassifier(num_classes)
        self.phase_angle_model = PhaseAngleLayerClassifier(num_classes)
        self.comod_model = ComodLayerClassifier(num_classes)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, pmag, pmag_mask, pang, pang_mask, comod, comod_mask):
        """
        Forward pass through the ensemble.
        
        Args:
            pmag (torch.Tensor): Phase magnitude input
            pmag_mask (torch.Tensor): Phase magnitude mask
            pang (torch.Tensor): Phase angle input
            pang_mask (torch.Tensor): Phase angle mask
            comod (torch.Tensor): Comodulation input
            comod_mask (torch.Tensor): Comodulation mask
            
        Returns:
            torch.Tensor: Output logits
        """
        # Get logits from each model
        pmag_logits = self.phase_mag_model(pmag, pmag_mask)
        pang_logits = self.phase_angle_model(pang, pang_mask)
        comod_logits = self.comod_model(comod, comod_mask)
        
        # Concatenate logits
        combined = torch.cat([pmag_logits, pang_logits, comod_logits], dim=1)
        
        # Final fusion
        ensemble_logits = self.fusion(combined)
        
        return ensemble_logits
