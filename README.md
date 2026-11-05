<h1 align="center">
  <img src="assets/roar_icon.png" width="100" />
  ROAR
</h1>

<p align="center">
  <img src="./assets/tbs.gif" width="48%">
  <img src="./assets/rts.gif" width="48%">
</p>

<p align="center">
<em>These two videos compare the behavior of Gemini-3.1 Pro in the basic single-agent level (goal: collect three coins) under the turn-based setting (left) and the real-time setting (right) with a fixed interval of 40 ms. Under the turn-based setting, the agent can avoid turret attacks through timely movement, whereas under the real-time setting, it is hit by the turret.</em>
</p>

<p align="center">
  <img src="./assets/tb.gif" width="48%">
  <img src="./assets/rt.gif" width="48%">
</p>

<p align="center">
<em>These two videos compare the behaviors of Gemini-3.1 Pro (top two, control the knight) and GPT-5.4 (bottom two, control the clown) in the mixed level (2v2 team battle, goal: defeat all opponents) under the turn-based setting (left) and the real-time setting (right) with a fixed interval of 40 ms. Under the turn-based setting, Gemini-3.1 Pro quickly defeats GPT-5.4 due to its stronger performance. In contrast, under the real-time setting, Gemini-3.1 Pro is eliminated by the toxic zone due to delayed actions.</em>
</p>



# 📖 How to run
## 1. Unzip the executable file for the corresponding ROAR environment level in `./environments`. You may also try it out yourself.😊

## 2. Install the Conda environment using the `./environment.yaml` file with Python version 3.10.12.

## 3. Fill in the path to the environment executable unzipped in Step 1 and the path to the configuration file in `roar_evaluation/const_string.py`, following the locations indicated in this file. Also configure the API keys for the LMMs in this file. For easier reproducibility, all API calls have been adapted to use OpenRouter by default, but you may modify the API interfaces in the code as needed.

## 4. Run `roar_evaluation/roar_benchmark.py` to start the experiment. You may modify the parameters in the script to conduct different experiments.

# 🔥 This repository is actively being updated. Stay tuned for more!