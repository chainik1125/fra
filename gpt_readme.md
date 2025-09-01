North-star: Show that Feature-Resolved Induction (FR-IS) gives a more causal and compact explanation of induction heads than standard token-level attention attribution—on a tiny, reproducible benchmark. Induction heads + repeated-sequence prompts are the canonical setup for this.

https://arxiv.org/abs/2209.11895?utm_source=chatgpt.com
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html?utm_source=chatgpt.com

Classic interpretability inspects token→token attention maps. But raw attention weights can be unreliable as explanations; stronger token-level baselines like attention rollout/flow improve faithfulness but still operate at the token level. FRA asks: what if we resolve tokens into SAE features and analyze attention at the feature-pair level instead? If the top feature pairs are sparser and more causal than token selections, FRA is a better tool for this use case.

