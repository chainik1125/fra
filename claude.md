Project goals

The aim of the project is to demonstrate a new approach to attention heads in mechanistic interpretability that uses Sparse Autoencoder features (SAEs) in a new way.
The new approach is called "Feature-Resolved Attention" (FRA). The core idea is very simple. Instead of looking at [context,context] sized attention patterns that show the attention between tokens,
we want to "resolve" the tokens into their SAE features so that we now have a "Feature-Resolved" attention map of size [context,context,hidden_dim,hidden_dim] where hidden dimension is the SAE hidden dimension. We will then attempt to demonstrate the Feature-Resolved attention gives more insight than standard attention maps, and hence provide a new tool for the community.

This is a speculative, very-high speed project that is aiming to show a quick, flashy result that shows signs of life within 10h of total work (dev + runtime) so we must prioritize rough results over precision. Despite that, we cannot allow any serious technical flaws - it is much better to correctly show that this is not interesting than to incorrectly show that it is.



The current project plan is:

1. Choose an SAE, model, and dataset that are:
    a. Are quickly runnable on an a100 gpu.
    b. The SAE must have interesting, clearly interpretable features. Ideally the SAE should be trained on the part of residual stream immediately before attention. In GPT-2 this is ln1.hook_normalized.
    c. The Model and dataset must have been previously studied and induction heads found - this is crucial for stage 3!
    d. That then entails that we must have the SAE in those layers where the attention head was found.
    If the model does not use Layernorm that is an advantage.
    c. The model should use attention (pre-softmax) in a standard way.
    d. Everything must use standard tooling as far as possible (transformer-lens, sae-lens, neuronpedia etc....)
    f. Most (not all) meaningful text samples should be around 128 tokens, up to 256 is fine.

2. Demonstrate feature resolved attention on a single text sample.
    a. I have already implemented this before in a mean-subtracting crosscoder. The function is defined at the bottom as analyze_feature_attention_interactions.
    This returns the FA[i,j]=[hidden_dim,hidden_dim] object for a given query and key position in an input sample of text. Simple computing this for the n(n+1)/2 qk sequence position
    pairs gives the FA=[context,context,hidden_dim,hidden_dim] object we want. My key concern here is how to handle the memory issues. Since everything is sparse, it is definetly doable
    I'm just not sure what the best implementation is. In the current implementation we compute everything sparsely but then broadcast back to the original shape. Probably the simplest thing
    is just to keep the final broadcast back in a sparse tensor object, but we should think about this.

	
    b. The first port-of-call is to compute this for some example sentences, and see if we get interesting feature-interactions.
        i. How will we search for them? Not sure - obvious things are a. the average largest values when the feature activation is non-zero: topk over i,j (FA[:,:,i,j].sum()/couint(FA[:,:,i,j]>0)) if FA
        could be b. topk across the sample anywhere topk(FA), might filter for non-punctuation things etc...
        ii. It may be good to come up with sentences where you expect feature interaction - maybe in coding contexts? Other contexts? Is there literature here either in attention maps (IoI etc) or in 
        feature interaction attribution literature?
    
    c. Pause at this point for 30m of searching for efficiency quick wins, but no longer than this unless this is a big bottleneck.

3. The key test I want to do is to try to find "Conceptual Induction Heads" which is what I'm calling an induction head within a feature-resolved channel i.e. an induction head when resolved into a feature-pair channel FA[:,:,i,j]. The baseline here is token-token attention heads. My key hypothesis is that CIH > Token Induction Heads (TIH) - i.e. we will be able to find induction head feature channels for heads that are not token induction heads, and conversely that (roughly) every Token Induction Head will have at least one conceptual induction head. 
    a. We will do both tests. Start from known TIH (note - you must have chosen a model and dataset in which this is true in stage 1 for this to work!) and do the induction head test in tranformerlens for CIH.
    UNCERTAINTY: What dataset text/samples do we need to do this?
    b. Then the converse - search over heads and try to find heads that are not TIH but which nonetheless have CIH channels. 
    c. Can we understand the channels as their own unit of analysis - independent of tokens.
4. The stretch application is to try to say something interesting about reasoning models and/or Chain of Thought. Can we apply this feature resolved perspective to look at sentences? Can we do the Feature-Resolved version of thought anchors? Maybe we could come back to the idea of measuring the internal consistency of a CoT through this?

    
    

    









####Reference functions - this codebase

####Reference functions - from previous (not adapted to this codebase)

def analyze_feature_attention_interactions(self, layer: int, head: int, 
											 input_text: str, query_position: int, key_position: int):
		"""
		Analyze interactions between features in attention.
		
		Args:
			layer: Layer index
			head: Head index
			input_text: Input text to analyze
			query_position: Query position
			key_position: Key position
		"""
		activations_SMPD = self.get_llm_activations(input_text, subtract_mean=True)
		feature_activations_SH = self.crosscoder._encode_BH(activations_SMPD)
		
		feature_activations_query = feature_activations_SH[query_position]
		query_active_features = torch.where(feature_activations_query != 0)[0]
		
		feature_activations_key = feature_activations_SH[key_position]
		key_active_features = torch.where(feature_activations_key != 0)[0]
		
		query_activations_for_features = self.crosscoder.W_dec_HXD[
			query_active_features, 0, self.hookpoints.index(f"blocks.{layer}.ln1.hook_normalized")
		]
		key_activations_for_features = self.crosscoder.W_dec_HXD[
			key_active_features, 0, self.hookpoints.index(f"blocks.{layer}.ln1.hook_normalized")
		]
		
		interaction_matrix_unscaled = self.attention_pattern_QK(
			layer, head,
			query_activations_for_features, False,
			key_activations_for_features, False
		)
		
		query_active_features = query_active_features.cpu().numpy()
		key_active_features = key_active_features.cpu().numpy()
		
		matrix_scaling = feature_activations_query[query_active_features].unsqueeze(1) * \
						feature_activations_key[key_active_features].unsqueeze(0)
		matrix_scaling = matrix_scaling.to("cpu").numpy()
		
		if self.feature_activations_active_mean is not None:
			interaction_matrix_unscaled *= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
			interaction_matrix_unscaled *= self.feature_activations_active_mean[key_active_features][np.newaxis, :]
			matrix_scaling /= self.feature_activations_active_mean[query_active_features][:, np.newaxis]
			matrix_scaling /= self.feature_activations_active_mean[key_active_features][np.newaxis, :]
		
		return {
			'query_active_features': query_active_features,
			'key_active_features': key_active_features,
			'interaction_matrix_unscaled': interaction_matrix_unscaled,
			'matrix_scaling': matrix_scaling,
			'interaction_matrix': interaction_matrix_unscaled * matrix_scaling
		}
        

        	def attention_pattern_QK(self, layer: int, head: int, q_input: torch.Tensor, 
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
		W_Q = self.llm.blocks[layer].attn.W_Q[head]
		b_Q = self.llm.blocks[layer].attn.b_Q[head]
		W_K = self.llm.blocks[layer].attn.W_K[head]
		b_K = self.llm.blocks[layer].attn.b_K[head]
		
		q = einsum(W_Q, q_input, "d a, s d -> s a")
		if q_do_bias:
			q += b_Q
			
		k = einsum(W_K, k_input, "d a, s d -> s a")
		if k_do_bias:
			k += b_K
			
		attention_scores = einsum(q, k, "q a, k a -> q k")
		return attention_scores.to("cpu").numpy()