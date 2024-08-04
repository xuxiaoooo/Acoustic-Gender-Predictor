import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

class CNNAttentionModel(nn.Module):
    def __init__(self, input_size, fixed_output_size=512, num_classes=2):
        super(CNNAttentionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.fc1 = nn.Linear(fixed_output_size, 128)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        
        if x.size(1) != 512:
            x = F.adaptive_avg_pool1d(x.unsqueeze(1), 512).squeeze(1)
        
        x = self.fc1(x)
        
        x = x.unsqueeze(0).permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.mean(dim=1)
        output = self.fc2(attn_output)
        return F.log_softmax(output, dim=1)


class GenderAudioPredictor:
    def __init__(self, model_path, fixed_length=256):
        self.fixed_length = fixed_length
        self.model = CNNAttentionModel(input_size=fixed_length)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def _pad_or_truncate(self, spectrogram, max_length):
        if spectrogram.shape[-1] < max_length:
            pad_size = max_length - spectrogram.shape[-1]
            spectrogram = nn.functional.pad(spectrogram, (0, pad_size))
        else:
            spectrogram = spectrogram[:, :, :max_length]
        return spectrogram
    
    def predict(self, audio_file):
        waveform, sample_rate = torchaudio.load(audio_file)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
        mel_spectrogram = self._pad_or_truncate(mel_spectrogram, self.fixed_length)
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(mel_spectrogram)
            _, predicted = torch.max(outputs, 1)
        
        return 'male' if predicted.item() == 0 else 'female'

# Usage example:
model_path = 'audio_gender.pth'
predictor = GenderAudioPredictor(model_path)

audio_file = 'audio.wav'
gender = predictor.predict(audio_file)
print(f'The predicted gender for the audio file is: {gender}')