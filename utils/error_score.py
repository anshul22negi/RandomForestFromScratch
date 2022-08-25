def error_score(prediction, ground_truth):
    mis_label = 0
    for i in range(len(prediction)):
        if prediction[i] != ground_truth[i]:
            mis_label += 1
    return mis_label / len(prediction)