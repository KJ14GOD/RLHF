 detailed pipeline
 
1. policy model - llm i am training with ppo
 2 reference model - frozen copy of llm for kl divergence
 3 reward model - a model trained to score responses as better or worse using Anthropic dataset
4 value head - a small head on top of the policy model that predicts expected reward for PPO

rlhf has 3 main components
1. train reward model
2. generate responses from policy model
 3. use ppo to improve policy using reward model scores

policy model generates a response
 reward model scores the response
# ppo updates the policy model to make high scoring responses more likely
# reference model is used to calculate kl divergence so that it doesn't go too far from the reference model 
