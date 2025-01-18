# 🏔️ TIBET for Bias Detection

#### **[ECCV 2024] TIBET: Identifying and Evaluating Biases in Text-to-Image Generative Models**

*Aditya Chinchure🔸, Pushkar Shukla🔸, Gaurav Bhatt,  Kiri Salij,  Kartik Hosanagar, Leonid Sigal, Matthew Turk*
(🔸 indicates equal contribution)

### News
- 🌟 Accepted to ECCV 2024!
- 📚 Paper available on [arXiv](https://arxiv.org/abs/2312.01261)

### Dataset
The dataset is available to download here: [TIBET Website](https://tibet-ai.github.io). This dataset contains 100 prompts, and their associated bias axes, counterfactuals, and 48 images for each prompt and counterfactual generated using Stable Diffusion 2.1.

### Code

This codebase is based on the [MiniGPTv2](https://github.com/Vision-CAIR/MiniGPT-4) codebase. We set this up with Stable Diffusion 2.1 to generate images (see the `tibet/gen_images.py` file). You may choose to use any model you like. Finally, TIBET requires you to use your own OpenAI API key (see step 4) to obtain counterfactual prompts. 

1. Environment setup:
```
conda create --name tibet python=3.11
conda activate tibet
conda install cudatoolkit
pip install -r requirements.txt
```

2. Update `ckpt` in `vqa_configs/minigptv2_eval.yaml` to the path of the model checkpoint. The model checkpoint can be downloaded from [here](https://github.com/Vision-CAIR/MiniGPT-4). We use the `MiniGPT-v2 (online developing demo)` configuration.

3. [Maybe optional] Following the instructions in the [MiniGPTv2 Repo](https://github.com/Vision-CAIR/MiniGPT-4), you also need to download Llama 2 weights and update the path in `minigpt4/configs/models/minigpt_v2.yaml`. It may also be possible to let Huggingface handle this, as long as you have signed into Huggingface CLI and have access to the Llama 2 7B model.

4. Update TIBET configurations in `main.py`:
- `NUM_IMAGES = 2` - Number of images to generate per prompt and counterfactural prompt. Default in the paper is 48 images
- `IMG_DIR = 'images'` - Base directory path to save images
- `OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')` - OpenAI API key
- `PROMPT = "A photo of a computer programmer"` - Prompt to conduct bias analysis on
- `PROMPT_ID = 'P_1'` - Prompt ID, which is the folder name for the images and json files. Useful when you have multiple prompts to deal with.

5. Run the code:
```
python main.py
```

### Citing our work
```
@inproceedings{chinchure2025tibet,
  title={TIBET: Identifying and evaluating biases in text-to-image generative models},
  author={Chinchure, Aditya and Shukla, Pushkar and Bhatt, Gaurav and Salij, Kiri and Hosanagar, Kartik and Sigal, Leonid and Turk, Matthew},
  booktitle={European Conference on Computer Vision},
  pages={429--446},
  year={2025},
  organization={Springer}
}
```
