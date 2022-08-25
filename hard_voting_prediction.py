from sklearn import tree

def hard_voting_prediction(test_sample, forest):
    predictions_list = list()
    for i in range(len(test_sample)):
        predictions = dict()
        predictions[0]=0
        predictions[1]=0
        predictions_list.append(predictions)
    for decision_tree in forest:
        prediction = decision_tree.predict(test_sample)
        for i in range(len(prediction)):
            predictions_list[i][prediction[i]]+=1
    
    final_pred = list()
    for predictions in predictions_list:
        final_pred.append(max(predictions, key= lambda x: predictions[x]))
    
    return final_pred

