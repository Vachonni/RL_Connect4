# RL_Connect4
RL for Connect 4

Possible to train with Google Colab


WHY???
Need to `run pip install . -e` in you env to have access to local folders, even if called in Dockerfile.

OBSERVATIONS:
- LearnToStop implies forgetting old adversaries. Better than last, but less when compare to inital models
- 1000 can beat 5000. Is 1000 iterations enough?
- First models on a serie are easier to beat, then more complicated, closer to equal even if continue to train. Is this because model has not enough capacity?
- Rewards in Play are only 1 or -1 at the end. Is this the case while traiing or 1, -10 and 1/42?

NEXT STEPS:
- Try with a model that has more capacity (SEE OBSERVATIONS)
- Keep learning from all (?) previous agents (to avoid catastrophic forgetting) (SEE OBSERVATIONS)
- Config as YAML (ex: https://datacrayon.com/practical-evolutionary-algorithms/yaml-for-configuration-files/)
- Delete KaggleLearn (not matching ModelpathToAgent anymore)


