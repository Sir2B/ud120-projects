#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    for index in range(len(predictions)):
        error = predictions[index][0] - net_worths[index][0]
        error *= error
        cleaned_data.append((ages[index][0], net_worths[index][0], error))
    
    cleaned_data.sort(key=lambda x: x[2]) 
    return cleaned_data[:int(len(predictions)*0.9)]

