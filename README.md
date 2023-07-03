# RL_Connect4

Possible to train with Google Colab, 
but models can't to transfered (serialization issus known by StableBaselines)

## Connection to Azure ML

Connection information to Worspace are in devcontainer.env

<br><br><br>

#### Observations:
- LearnToStop implies forgetting old adversaries. Better than last, but less when compare to inital models
- 1000 can beat 5000. Is 1000 iterations enough?
- First models on a serie are easier to beat, then more complicated, closer to equal even if continue to train. Is this because model has not enough capacity?
- Rewards in Play are only 1 or -1 at the end. Is this the case while traiing or 1, -10 and 1/42?

#### Next steps:
- Keep learning from all (?) previous agents (to avoid catastrophic forgetting) (SEE OBSERVATIONS)
- Delete KaggleLearn (not matching ModelpathToAgent anymore)


