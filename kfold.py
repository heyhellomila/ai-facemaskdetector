

# Reference for k fold was obtained here: https://stackoverflow.com/questions/36063014/what-does-kfold-in-python

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
i = 0

kfold = KFold(10, True, 1)

for train, test in kfold.split(original_set):
    i += 1
    num_epochs = 25
    num_classes = 4
    learning_rate = 0.001

    model = CNN()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(original_set[train])
    loss_list = []
    acc_list = []
    iterations = []

    n = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(original_set[train]):
            # Move them to device (cuda or cpu)
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Saving the trained model and emptying the GPU memory
    torch.save(model.state_dict(), 'file path')
    # del model
    torch.cuda.empty_cache()

    # Loading the model before testing
    device = 'cpu'
    model = CNN()
    model.load_state_dict(torch.load('file path'))
    model = model.to(device)

    # overall metrics
    for pred_images, pred_labels in original_set[test]:
        pred_images = pred_images.to(device)
        pred_labels = pred_labels.to(device)
        pred_outputs = model(pred_images)
        _, pred_predicted = torch.max(pred_outputs.data, 1)

    accuracy_i = accuracy_score(pred_labels, pred_predicted)
    precision_i = precision_score(pred_labels, pred_predicted)
    recall_i = recall_score(pred_labels, pred_predicted)
    f1_i = f1_score(pred_labels, pred_predicted)

    print('The metrics for fold no. ', i)

    print('Accuracy = ', accuracy_i)
    print('Precision = ', precision_i)
    print('Recall = ', recall_i)
    print('f1 = ', f1_i)
    print()

    accuracy_list.append(accuracy_i)
    precision_list.append(precision_i)
    recall_list.append(recall_i)
    f1_list.append(f1_i)

print()
print('The overall performance metrics over all 10 folds are as follows:')
print()
print('Accuracy = ', sum(accuracy_list) / 10)
print('Precision = ', sum(precision_list) / 10)
print('Recall = ', sum(recall_list) / 10)
