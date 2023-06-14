import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

torch.__version__

# Creating a straight line dataset
weight = 0.3
bias = 0.9
start = 0
end = 1
step = 0.01

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

len(X)

train_split = int(0.8 * len(X))

train_X, train_y = X[:train_split], y[:train_split]
test_X, test_y = X[train_split:], y[train_split:]

len(train_X), len(train_y), len(test_X), len(test_y)

def plot_pred(train_data=train_X, train_label=train_y, test_data=test_X, test_label=test_y, predictions=None):
  plt.figure(figsize=(15, 7))

  plt.scatter(train_data, train_label, c="b", s=4, label="Training data")
  plt.scatter(test_data, test_label, c="r", s=4, label="Testing data")

  if predictions is not None:
    plt.scatter(test_data, predictions, c="g", s=4, label="Predictions")

  plt.legend(prop={"size":15});

plot_pred()

class LinearModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
    self.bias = nn.Parameter(torch.randn(1,
                                          requires_grad=True,
                                          dtype=torch.float))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weights * x + bias


model = LinearModel()

model.state_dict()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.001)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Check it with inference_mode()
with torch.inference_mode():
  preds_y = model(test_X)

preds_y

plot_pred(predictions=preds_y)

# Training and testing loop

epochs = 10000
epoch_count = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):
  ### Training
  model.train()

  pred_y = model(train_X)

  loss = loss_fn(pred_y, train_y)

  optimizer.zero_grad()

  loss.backward()

  optimizer.step()

  ### Testing
  with torch.inference_mode():
    test_pred = model(test_X)

    test_loss = loss_fn(test_pred, test_y.type(torch.float))

  if epoch % 1000 == 0:
    epoch_count.append(epoch)
    train_loss_values.append(loss.detach().numpy())
    test_loss_values.append(test_loss.detach().numpy())
    print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss}")

model.state_dict()



with torch.inference_mode():
  preds_y = model(test_X)

plot_pred(predictions=preds_y)

# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();

test_loss_values

from pathlib import Path

model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

model_name = "linear_regression_exercise.pth"
model_save_path = model_path / model_name

torch.save(obj=model.state_dict(), f=model_save_path)

loaded = LinearModel()

loaded.load_state_dict(torch.load(f=model_save_path))

loaded.eval()
with torch.inference_mode():
  loaded_preds = loaded(test_X)

loaded_preds

plot_pred(predictions=loaded_preds)



