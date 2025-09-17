import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader
import warnings
import random
import torch.nn as nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

warnings.filterwarnings('ignore')

scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            nn.Dropout(dropout))

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features))

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads=4):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        N = x.shape[0]
        x = x.view(N, self.heads, self.head_dim)

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        energy = torch.einsum("nhd,nhd->nh", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=1)

        out = torch.einsum("nh,nhd->nhd", [attention, values])
        out = out.reshape(N, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out

class AdvancedMultiTaskModel(nn.Module):
    def __init__(self, input_dim, n_classes, dropout=0.3):
        super(AdvancedMultiTaskModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(256, 128, dropout=dropout),
            ResidualBlock(128, 64, dropout=dropout))

        self.attention = MultiHeadAttention(64)

        self.age_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1))

        self.class_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes))

    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features = self.attention(features)
        age_output = self.age_head(attended_features)
        class_output = self.class_head(attended_features)
        return age_output, class_output

def preprocess_test_data(data_path):
    data = pd.read_excel(data_path)
    print(f"测试数据形状: {data.shape}")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns

    for col in data.columns:
        if col in numeric_cols:
            mask = data[col].isna()
            median_val = data[col].median()
            random_noise = np.random.normal(0, data[col].std() * 0.1, size=mask.sum())
            data.loc[mask, col] = median_val + random_noise
        elif col in categorical_cols:
            data[col] = data[col].fillna("UNK")

    for col in categorical_cols:
        freq = pd.Series(scaler.mean_, index=scaler.feature_names_in_).filter(like=f'{col}_freq_enc').values
        if len(freq) > 0:
            data[f'{col}_freq_enc'] = data[col].map(lambda x: freq[0] if x not in locals() else locals()[x])

    feature_cols = [col for col in scaler.feature_names_in_ if col in data.columns]
    data[feature_cols] = scaler.transform(data[feature_cols])

    X = data.drop(['age'], axis=1, errors='ignore')
    X = X[scaler.feature_names_in_]

    return X

def predict(data_path, model_path='best_model.pth'):
    X_test = preprocess_test_data(data_path)

    n_classes = len(label_encoder.classes_)

    model = AdvancedMultiTaskModel(input_dim=X_test.shape[1], n_classes=n_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_tensor = torch.FloatTensor(X_test.values).to(device)
    test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_age_preds = []
    all_class_preds = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            age_output, class_output = model(inputs)
            print(age_output, class_output)

            age_pred = np.round(age_output.cpu().numpy()).astype(int)
            all_age_preds.extend(age_pred.flatten())

            class_pred = torch.argmax(class_output, dim=1).cpu().numpy()
            all_class_preds.extend(class_pred)

    class_pred_labels = label_encoder.inverse_transform(all_class_preds)

    results = pd.DataFrame({
        '预测年龄': all_age_preds,
        '预测类别': class_pred_labels
    })

    original_data = pd.read_excel(data_path)
    final_results = pd.concat([original_data, results], axis=1)

    return final_results

if __name__ == "__main__":
    test_data_path = "2.xlsx"

    print("开始预测...")
    predictions = predict(test_data_path)

    predictions.to_excel("预测结果2.xlsx", index=False)
    print("预测完成，结果已保存至 '预测结果2.xlsx'")
    print(predictions)