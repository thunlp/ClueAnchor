# ClueAnchor: Clue-Anchored Knowledge Reasoning Exploration and Optimization for Retrieval-Augmented Generation

<p align="center">
  <a href="https://github.com/thunlp/ClueAnchor" alt="GitHub">
    <img src="https://img.shields.io/badge/GitHub-ClueAnchor-black?logo=github"/>
  </a>
  <a href="https://arxiv.org/abs/2505.24388" alt="Paper">
    <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?logo=arxiv&logoColor=white"/>
  </a>
  <a href="https://huggingface.co/MethaneChen222/ClueAnchor" alt="Model">
    <img src="https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface"/>
  </a>
  <a href="https://huggingface.co/datasets/MethaneChen222/ClueAnchor" alt="Dataset">
    <img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface"/>
  </a>
</p>

This repository is the official implementation of [ClueAnchor: Clue-Anchored Knowledge Reasoning Exploration and Optimization for Retrieval-Augmented Generation](https://arxiv.org/abs/2505.24388).

![ClueAnchor](/assets/method.png)

## ⚙️ Requirement

Install the following packages using Pip or Conda under this environment:

```txt
python==3.10.14
torch==2.4.0
transformers==4.46.0
trl==0.12.0
vllm==0.5.5
accelerate==1.4.0
deepspeed==0.16.3
peft==0.12.0
```



## 🔎 ClueAnchor Pipeline

### 🗂️ Prepare the Training and Test Data

#### (1) Clone the Repository

Use the following command to clone the project:

```bash
git clone https://github.com/thunlp/ClueAnchor
cd ClueAnchor
```

#### (2) Construct `original train/dev/test dataset`

To ensure consistency and compatibility across downstream modules, all raw data should be converted into a standardized JSON format.
 **Each sample must contain the following four required fields:**

```json
{
  "id": "A unique identifier for the sample (int)",
  "question": "The input question (str)",
  "answer": "The ground truth answer to the question (str)",
  "data_type": "The dataset or task name, e.g., 'NQ', '2WikiMQA', etc. (str)"
}     
```

#### (3) Construct `retrieval-augmented dataset`

The retrieval step in this project is partially adapted from the [RAG-DDR](https://github.com/OpenBMB/RAG-DDR) project. We reuse several utility scripts with minor modifications to support our own pipeline.

We use the [`bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) embedding model to retrieve relevant passages from an English Wikipedia corpus, which is based on the version released by [Izacard et al. (2022b)](https://arxiv.org/abs/2208.03299v1). These retrieved passages are then integrated into the training and development data to build retrieval-augmented datasets.

##### 🔧 Step-by-Step Instructions:

**(a) Encode the document corpus into embeddings**

Navigate to the `retrieval` folder and run:

```bash
cd retrieval
bash getembedding.sh
```

📌 *This script uses `bge-large-en-v1.5` to encode all documents and saves the embeddings to the directory specified by `--output_dir`.*

**(b) Retrieve relevant documents for each query**

Once the embeddings are ready, run:

```bash
bash retrieve.sh
```

📌 *This script retrieves relevant passages for each query from the `--query_path` file and saves the results in TREC format to `--trec_save_path`.*

**(c) Construct the final retrieval-augmented dataset**

Finally, build the augmented train/dev dataset by merging the retrieved documents with the original data:

```bash
bash construct_psg_data.sh
```

📌 *The output will be saved to `--output_path` as a new version of the train/dev set that includes the top-ranked retrieved passages.*



### 🕵️ Knowledge Reasoning Exploration

After constructing the training and test data, you can proceed to build the **Knowledge Reasoning Exploration Module**.
In this stage, the model is encouraged to explore different reasoning paths to arrive at the correct answer. This helps improve its ability to handle complex, multi-hop, or ambiguous questions by simulating various reasoning strategies. You need to download [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model as the vanilla Generation Model.

We support three types of knowledge reasoning exploration, You can either follow the steps sequentially, using the output of each step as the input to the next, execute them separately and merge their results into a single file at the end.

#### (1) Internal Knowledge Reasoning

This mode encourages the model to reason using only its internal knowledge, without access to external documents.

```bash
cd scripts
bash Internal_Knowledge_Reasoning.sh
```

#### (2) External Knowledge Reasoning

In this mode, the model is provided with retrieved passages from an external knowledge source (e.g., Wikipedia) to support its reasoning process.

```bash
bash Internal_Knowledge_Reasoning.sh
```

📌 *Note: Please ensure you have already run the retrieval step and have access to document embeddings and retrieved results.*

#### (3) Clue-Anchored Knowledge Reasoning

This approach introduces an intermediate clue extraction step. The model first identifies key clues in the retrieved passages and then uses them to guide its reasoning path.

```bash
bash Clue_Extraction.sh
bash ClueAnchored_Knowledge_Reasoning.sh
```

📌*Clue-anchored reasoning can help the model focus on relevant information early in the reasoning process, leading to more accurate and interpretable results.*



### 🛠️ Knowledge Reasoning Optimization

After generating diverse reasoning paths, we further optimize the model’s ability to select and utilize knowledge through two key steps: **Reward-guided Knowledge Selection** and **Knowledge Preference Optimization**.

#### (1) Reward-guided Knowledge Selection

This step filters and ranks the generated reasoning paths based on a reward signal. The goal is to select the most promising knowledge paths for training the final model.

```bash
cd scripts
bash Knowledge_Selection.sh
```

#### (2) Knowledge Preference Optimization

This step fine-tunes the model using preference-based learning techniques, encouraging it to prefer high-quality knowledge reasoning paths over less effective ones. 

```bash
bash Knowledge_preference_optimization.sh
```

📌 *This script takes the filtered high-quality paths as input and performs preference optimization to further align the model's behavior with desirable reasoning strategies.*



## 🎯 Evaluation

After completing the full ClueAnchor pipeline and training your model, you can evaluate its performance on various downstream tasks using the provided evaluation scripts.

Navigate to the `eval` directory and execute the evaluation script:

```bash
cd scripts
bash Eval.sh
```

This script will automatically run evaluation on the trained model across different benchmark tasks, generating metrics such as accuracy, or other task-specific scores. The evaluation results will be saved in the specified output directory for further analysis.



## 📖 Citation

If you find this work useful, please cite our paper and give us a shining star 🌟

```
@article{chen2025clueanchor,
  title={ClueAnchor: Clue-Anchored Knowledge Reasoning Exploration and Optimization for Retrieval-Augmented Generation},
  author={Chen, Hao and Yan, Yukun and Mei, Sen and Che, Wanxiang and Liu, Zhenghao and Shi, Qi and Li, Xinze and Fan, Yuchun and Huang, Pengcheng and Xiong, Qiushi and others},
  journal={arXiv preprint arXiv:2505.24388},
  year={2025}
}
```



## ✉️ Contact

If you have questions, suggestions, and bug reports, please email:

```
methanechen@126.com
```


