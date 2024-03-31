import pdb

def dict_weight(dict1, weight):
    for k, v in dict1.items():
        dict1[k] = weight * v
    return dict1
    
def dict_add(dict1, dict2):
    for k, v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1


def FedAvg(global_model, local_models, client_weight):
    new_model_dict = None

    for client_idx in range(len(local_models)):
        local_dict = local_models[client_idx].state_dict()

        if new_model_dict is None:  # init
            new_model_dict = dict_weight(local_dict, client_weight[client_idx])
        else:
            new_model_dict = dict_add(new_model_dict, dict_weight(local_dict, client_weight[client_idx]))

    global_model.load_state_dict(new_model_dict)


def FedUpdate(global_model, local_models):
    global_dict = global_model.state_dict()

    for client_idx in range(len(local_models)):
        local_models[client_idx].load_state_dict(global_dict)
    
