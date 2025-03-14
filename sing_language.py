import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 26)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = torchvision.datasets.ImageFolder(root='C:/path/to/your/dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

train_model(model, train_loader, criterion, optimizer)

torch.save(model.state_dict(), 'sign_language_model.pth')

cap = cv2.VideoCapture(0)

def preprocess_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = np.array(frame) / 255.0
    frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()
    return frame.to(device)

model.load_state_dict(torch.load('sign_language_model.pth'))
model.eval()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_image = preprocess_image(frame)
    with torch.no_grad():
        output = model(input_image)
    predicted_class = torch.argmax(output, 1).item()

    predicted_letter = chr(predicted_class + 65)

    cv2.putText(frame, f'Predicted: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
