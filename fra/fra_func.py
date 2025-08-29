from transformer_lens import HookedTransformer
import torch
import numpy as np
from typing import Any, Dict
from einops import einsum
from fra.activation_utils import get_llm_activations
from tqdm import tqdm


def lower_triangular_mask(pattern: np.ndarray) -> np.ma.MaskedArray:
    """Apply lower triangular mask to attention pattern."""
    mask = np.triu(np.ones(pattern.shape), k=1)
    return np.ma.array(np.tril(pattern, k=0), mask=mask)


def attention_pattern_QK(llm: Any, layer: int, head: int, q_input: torch.Tensor, 
                        q_do_bias: bool, k_input: torch.Tensor, k_do_bias: bool) -> np.ndarray:
    """
    Compute attention pattern from query and key inputs.
    
    Args:
        layer: Layer index
        head: Head index
        q_input: Query input tensor
        q_do_bias: Whether to add query bias
        k_input: Key input tensor
        k_do_bias: Whether to add key bias
        
    Returns:
        Attention scores as numpy array
    """
    W_Q = llm.blocks[layer].attn.W_Q[head]
    b_Q = llm.blocks[layer].attn.b_Q[head]
    W_K = llm.blocks[layer].attn.W_K[head]
    b_K = llm.blocks[layer].attn.b_K[head]
    
    q = einsum(W_Q, q_input, "d a, s d -> s a")
    if q_do_bias:
        q += b_Q
        
    k = einsum(W_K, k_input, "d a, s d -> s a")
    if k_do_bias:
        k += b_K
        
    attention_scores = einsum(q, k, "q a, k a -> q k")

    return attention_scores.detach().cpu().numpy()


def analyze_feature_attention_interactions(model: Any, sae: Any, layer: int, head: int, 
                                           input_text: str, query_position: int, key_position: int,
                                           hook_point: str = "hook_attn_out") -> Dict:
    """
    Analyze interactions between features in attention.
    
    Args:
        layer: Layer index
        head: Head index
        input_text: Input text to analyze
        query_position: Query position
        key_position: Key position
    """
    activations_SD = get_llm_activations(model,input_text,hook_point=hook_point,layers=layer)
    feature_activations_SH = sae.encode(activations_SD)
    
    feature_activations_query = feature_activations_SH[query_position]
    query_active_features = torch.where(feature_activations_query != 0)[0]
    
    feature_activations_key = feature_activations_SH[key_position]
    key_active_features = torch.where(feature_activations_key != 0)[0]
    
    query_activations_for_features = sae.W_dec[query_active_features]
    key_activations_for_features = sae.W_dec[key_active_features]
    
    interaction_matrix_unscaled = attention_pattern_QK(model, layer, head, 
                                                        query_activations_for_features, False, 
                                                        key_activations_for_features, False)


    
    # Convert to numpy after using for indexing
    query_features_tensor = query_active_features
    key_features_tensor = key_active_features
    
    matrix_scaling = feature_activations_query[query_features_tensor].unsqueeze(1) * \
                    feature_activations_key[key_features_tensor].unsqueeze(0)
    matrix_scaling = matrix_scaling.detach().cpu().numpy()
    
    query_active_features = query_features_tensor.cpu().numpy()
    key_active_features = key_features_tensor.cpu().numpy()
    
    # if self.feature_activations_active_mean is not None:
    #     interaction_matrix_unscaled *= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
    #     interaction_matrix_unscaled *= self.feature_activations_active_mean[key_active_features][np.newaxis, :]
    #     matrix_scaling /= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
    #     matrix_scaling /= self.feature_activations_active_mean[key_active_features][np.newaxis, :]
    
    return {
        'query_active_features': query_active_features,
        'key_active_features': key_active_features,
        'interaction_matrix_unscaled': interaction_matrix_unscaled,
        'matrix_scaling': matrix_scaling,
        'interaction_matrix': interaction_matrix_unscaled * matrix_scaling
    }


def get_sentence_averages(llm:Any,sae:Any,layer:int,head:int,input_text:str,hook_point:str="attn.hook_z"):
	text_length=128
	hidden_dim=sae.d_sae
	data_dep_int_matrix=np.zeros((hidden_dim,hidden_dim))
	data_dep_int_matrix_abs=np.zeros((hidden_dim,hidden_dim))
	data_dep_localization_matrix=np.zeros((hidden_dim,hidden_dim))
	count=0
	for key_index in tqdm(range(text_length), disable=True):
		for query_index in range(key_index,text_length):
				feature_analysis=analyze_feature_attention_interactions(llm,sae,layer,head,input_text,query_index,key_index,hook_point)
				int_matrix=feature_analysis["interaction_matrix"]
				query_active_features=feature_analysis["query_active_features"]
				key_active_features=feature_analysis["key_active_features"]
				data_independent=feature_analysis["interaction_matrix_unscaled"]
				
				resized_data_dependent_int=np.zeros((hidden_dim,hidden_dim))
				resized_data_dependent_int[query_active_features[:,None],key_active_features[None,:]]=int_matrix

				resized_data_dependent_localization=np.zeros((hidden_dim,hidden_dim))
				resized_data_dependent_localization[query_active_features[:,None],key_active_features[None,:]]=np.abs(int_matrix)*(query_index-key_index)

				
				
				
				data_dep_int_matrix=data_dep_int_matrix+resized_data_dependent_int
				data_dep_int_matrix_abs=data_dep_int_matrix_abs+np.abs(resized_data_dependent_int)
				data_dep_localization_matrix=data_dep_localization_matrix+resized_data_dependent_localization

				count+=1
	
	data_dep_int_matrix/=count
	data_dep_localization_matrix=data_dep_localization_matrix/np.clip(data_dep_int_matrix_abs,a_min=1,a_max=None)
	data_dep_int_matrix_abs/=count

	return data_dep_int_matrix,data_dep_int_matrix_abs,data_dep_localization_matrix

    


if __name__ == "__main__":
    print('main character')