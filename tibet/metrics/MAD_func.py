import numpy as np
from tibet.metrics.concept_matching import compute_association
from tibet.metrics.CAS_func import histogram_intersection_over_union
from tibet.metrics.clean_concepts import merge_clean_VQAans, get_freq_dict

#from scipy.special import softmax

def softmax(x, t=1):
    x = np.array(x)
    return np.exp(x/t)/sum(np.exp(x/t))

def MAD_func(data):
    """
    Calculate the Mean Absolute Deviation (MAD) of a given array of numbers.

    Parameters:
    data (array-like): The array of numbers.

    Returns:
    float: The Mean Absolute Deviation (MAD).
    """
    # Convert data to numpy array
    data_np = np.array(data)
    
    # Calculate the mean
    mean = np.mean(data_np)
    
    # Calculate the absolute deviations from the mean
    absolute_deviations = np.abs(data_np - mean)
    
    # Calculate the MAD
    mad = np.sqrt(np.mean(absolute_deviations))

    data_control = np.array([0]*(len(data_np)-1)+[1])
    mean_control = np.mean(data_control)
    absolute_deviations_control = np.abs(data_control - mean_control)
    mad_control = np.sqrt(np.mean(absolute_deviations_control))

    mad = mad/mad_control
    
    return mad


# Get the variance in CAS for a given axis of bias
def get_variance(p_dict, bias_axis, debug=False, do_per_VQA=True):

    num_images = len(p_dict['concepts_initial'])
    num_VQA = len(p_dict['concepts_initial'][0])

    # We compute scores in two ways,
    # (1) score per VQA question, by taking concepts across a single VQA
    # (2) overall score, by taking concepts across all VQA
    scores_per_VQA = []
    scores = []
    
    # (1) COMPUTE SCORES PER VQA
    if do_per_VQA:
        
        # For each VQA question
        for VQA_idx in range(num_VQA):

            main_concepts = merge_clean_VQAans(p_dict['concepts_initial'], VQA_idx=VQA_idx)
            C_init = get_freq_dict(main_concepts, num_images, threshold=0.01)

            VQA_concept_scores = []
            
            # For all counterfactuals
            for num_cf in range(len(p_dict['concepts_cf'][bias_axis])):

                cf_concepts = merge_clean_VQAans(p_dict['concepts_cf'][bias_axis][num_cf], VQA_idx=VQA_idx)
                C_cf = get_freq_dict(cf_concepts, num_images, threshold=0.01)
                
                print(VQA_idx, num_cf, C_init, C_cf)
                score = compute_association(C_init, C_cf, funcx=histogram_intersection_over_union)
                VQA_concept_scores.append(score)

            VQA_concept_scores = [score/sum(VQA_concept_scores) for score in VQA_concept_scores]

            scores_per_VQA.append(VQA_concept_scores)

        variance_per_VQA = [MAD_func(s) for s in scores_per_VQA]
        #variance_per_VQA = []
        # for scoresvqa in scores_per_VQA:
        #     score_rank_vqa = softmax([s*100 for s in scoresvqa])
        #     variance_vqa = wasserstein_distance(score_rank_vqa, [1/len(score_rank_vqa)]*len(score_rank_vqa))
        #     variance_per_VQA.append(variance_vqa)

    # (2) COMPUTE OVERALL SCORE
    main_concepts = merge_clean_VQAans(p_dict['concepts_initial'], VQA_idx=None)
    C_init = get_freq_dict(main_concepts, num_images, threshold=0.01)

    for num_cf in range(len(p_dict['concepts_cf'][bias_axis])):

        cf_concepts = merge_clean_VQAans(p_dict['concepts_cf'][bias_axis][num_cf], VQA_idx=None)
        C_cf = get_freq_dict(cf_concepts, num_images, threshold=0.01)

        score = compute_association(C_init, C_cf, funcx=histogram_intersection_over_union)
        scores.append(score)
    
    scores = [s/sum(scores) for s in scores]
    variance = MAD_func(scores)
    #scores = [score/sum(scores) for score in scores]
    #scores = [score for score in scores]
    #variance = np.std(scores)
    #print('using was')
    #print(scores)
    #print(score_rank)
    #variance = wasserstein_distance(scores, [1/len(scores)]*len(scores))
    #print('using range')
    #variance = max(scores) - min(scores)
    #print(scores)
    #print('mean:', np.mean(scores))
    # Get the rank order of the scores
    #score_rank=np.argsort(scores)
    #score_rank = scores + score_rank
    #score_rank = [score/sum(score_rank) for score in score_rank]
    

    if debug:
        print("CAS: ", scores)
        print("MAD: ", variance)
        print("Normalized Variance per VQA: ", [round(float(i)/sum(variance_per_VQA), 3) for i in variance_per_VQA])
        questions = p_dict['questions']
        for q,s in zip(questions, scores_per_VQA):
            print("Score for question: ", q, s)

    return variance, variance_per_VQA, scores, scores_per_VQA

def softmax_temp(x, tau):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: s -- 1-dimensional array
    """
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()