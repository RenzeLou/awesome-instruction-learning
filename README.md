<!-- omit in toc -->
# Awesome Textual Instruction Learning Papers


[![Awesome](https://awesome.re/badge.svg)](https://github.com/RenzeLou/awesome-instruction-learning) ![](https://img.shields.io/github/last-commit/RenzeLou/awesome-instruction-learning?color=green) ![](https://img.shields.io/badge/PaperNumber-0-blue) ![](https://img.shields.io/badge/PRs-Welcome-red)

A curated list of awesome **Instruction Learning** papers 🔥🔥🔥. 

Currently maintained by <ins>[Renze Lou](https://renzelou.github.io/) @ PennState</ins> and <ins>[Kai Zhang](https://drogozhang.github.io/) @ OhioState</ins>. 


**<font color='red'>Work still in progress</font>**  🚀, **we appreciate any suggestions and contributions** ❤️.

---

<!-- What is instruction learning?
Why instruction learning?
-->

<!-- TODO
## Our scope:
We aim to stay up-to-date with the most innovative developments in the field and gain valuable insights into the future of instruction-learning technology.👀 organize a systematic and comprehensive overview of instructional learning.

1. Stay up-to-date with the most innovative developments in this field.
2. Gain valuable insights into the future of instruction-learning technology.
3. 
-->


<!-- omit in toc -->
## How to contribute?

If you have any suggestions or find any missed papers, feel free to reach out or submit a [pull request](https://github.com/RenzeLou/awesome-instruction-learning/pulls):

1. Use following markdown format.

```markdown
**Paper Title.** *Author 1, Author 2, and Author 3.* <ins>Conference/Journal/Preprint</ins> Year. [[pdf](link)]; [[other resources](link)].
```
<!-- >1. **Paper Title.** *Author 1, Author 2, and Author 3.* Conference/Journal/Preprint Year. [[pdf](link)]. -->


2. If one preprint paper has multiple versions, please use **the earliest submitted year**.
   
3. Display the papers in a **year descending order**.


<!-- omit in toc -->
## Citation
Find this repository helpful? Please consider citing our paper:

```
work in progress.
```

---

<!-- omit in toc -->
## 🔍 Table of Contents 

- [1. 🎓 Surveys and Tutorials](#1--surveys-and-tutorials)
- [2. 🗂️ Taxonomies](#2-️-taxonomies)
  - [2.1 Entailment-oriented Instruction](#21-entailment-oriented-instruction)
  - [2.2 PLM-oriented Instruction](#22-plm-oriented-instruction)
  - [2.3 Human-oriented Instruction](#23-human-oriented-instruction)
- [3. 📊 Analyses](#3--analyses)
  - [3.1 Scale](#31-scale)
  - [3.2 Explanability](#32-explanability)
  - [3.3 Robustness](#33-robustness)
  - [3.4 Negation](#34-negation)
  - [3.5 Others](#35-others)
- [4. 🤖 Applications](#4--applications)
  - [4.1 Human-Computer Interaction](#41-human-computer-interaction)
  - [4.2 Data and Feature Augmentation](#42-data-and-feature-augmentation)
  - [4.3 General-purpose Language Models](#43-general-purpose-language-models)
- [5. 📚 Corpora](#5--corpora)
- [6. 🗒️ Other Papers](#6-️-other-papers)

---

## 1. 🎓 Surveys and Tutorials

<!-- There are several awesome surveys and tutorials on textual instruction learning. -->
<!-- To our knowledge, our survey is the first one to provide a comprehensive and broader overview of the field of instruction learning. -->
<!-- Since each survey focuses on specific in-context instruction, we attach a label to each of them to distinguish these topics.
, including `prompt`, `demonstrations`, `reasoning`, and `overview` (which means a broader perspective). -->


We use the label `comprehensive` to denote the papers with a more comprehensive perspective. While some other papers are more specific to a certain in-context instruction, including `prompt`, `demonstrations`, and `reasoning`.

1. **Is Prompt All You Need? No. A Comprehensive and Broader View of Instruction Learning.** *Renze Lou, Kai Zhang, and Wenpeng Yin.* <ins>Preprint</ins> 2023. [[pdf]()]; [[paper list](https://github.com/RenzeLou/awesome-instruction-learning)]. `comprehensive`.
   
2. **Learning from Task Instructions.** *Wenpeng Yin, Qinyuan Ye, Pengfei Liu, Xiang Ren, Hinrich Schütze.* <ins>Tutorial@EMNLP</ins> 2023. [[pdf]()]. `comprehensive`.

3. **Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing.** *Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig.* <ins>ACM Computing Surveys</ins> 2023. [[pdf](https://dl.acm.org/doi/pdf/10.1145/3560815)]; [[website](http://pretrain.nlpedia.ai/)]. `prompt`.
   
4. **A Survey on In-context Learning**. *Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui*. <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2301.00234.pdf)]. `demonstrations`.
   
5. **Towards Reasoning in Large Language Models: A Survey.** *Huang, Jie, and Kevin Chen-Chuan Chang.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2212.10403.pdf)]; [[paper list](https://github.com/jeffhj/LM-reasoning)]. `reasoning`.

6. **Reasoning with Language Model Prompting: A Survey.** *Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei Huang, and Huajun Chen.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2212.09597.pdf)]; [[paper list](https://github.com/zjunlp/Prompt4ReasoningPapers)]. `reasoning`.


## 2. 🗂️ Taxonomies

In our paper, we divide the textual instructions into three taxonomies.

<!-- TODO: add pic. -->

### 2.1 Entailment-oriented Instruction

   
1. **OpenStance: Real-world Zero-shot Stance Detection.** *Hanzi Xu, Slobodan Vucetic, and Wenpeng Yin.* <ins>CoNLL</ins> 2022. [[pdf](https://arxiv.org/pdf/2210.14299.pdf)]; [[code](https://github.com/xhz0809/OpenStance)].
   
2. **Ultra-fine Entity Typing with Indirect Supervision from Natural Language Inference.** *Bangzheng Li, Wenpeng Yin, and Muhao Chen.* <ins>TACL</ins> 2022. [[pdf](https://aclanthology.org/2022.tacl-1.35.pdf)]; [[code](link)]. 
   
3. **Textual Entailment for Event Argument Extraction: Zero- and Few-Shot with Multi-Source Learning.** *Oscar Sainz, Itziar Gonzalez-Dios, Oier Lopez de Lacalle, Bonan Min, and Eneko Agirre.* <ins>Findings of NAACL</ins> 2022. [[pdf](https://aclanthology.org/2022.findings-naacl.187.pdf)]; [[code](link)].

4. **Label Verbalization and Entailment for Effective Zero and Few-Shot Relation Extraction.** *Oscar Sainz, Oier Lopez de Lacalle, Gorka Labaka, Ander Barrena, and Eneko Agirre.* <ins>EMNLP</ins> 2021. [[pdf](https://aclanthology.org/2021.emnlp-main.92.pdf)]; [[code](link)].

5. **Adapting Language Models for Zero-shot Learning by Meta-tuning on Dataset and Prompt Collections.** *Ruiqi Zhong, Kristy Lee, Zheng Zhang, and Dan Klein.* <ins>Findings of EMNLP</ins> 2021. [[pdf](https://aclanthology.org/2021.findings-emnlp.244.pdf)]. 
   
6. **Incremental Few-shot Text Classification with Multi-round New Classes: Formulation, Dataset and System.** *Congying Xia, Wenpeng Yin, Yihao Feng, and Philip Yu.* <ins>NAACL</ins> 2021. [[pdf](https://aclanthology.org/2021.naacl-main.106.pdf)]; [[code]()].
   
7. **ExpBERT: Representation Engineering with Natural Language Explanations.** *Shikhar Murty, Pang Wei Koh, and Percy Liang.* <ins>ACL</ins> 2020. [[pdf](https://aclanthology.org/2020.acl-main.190.pdf)].
   
8.  **Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach.** *Wenpeng Yin, Jamaal Hay, Dan Roth* *.* <ins>EMNLP</ins> 2019. [[pdf](https://arxiv.org/pdf/1909.00161.pdf)].


### 2.2 PLM-oriented Instruction

We diaplay several representative works of PLM-oriented instruction learning (i.e., prompt learning). For more works, please refer to [this repo](https://github.com/thunlp/PromptPapers) and [this survey](https://dl.acm.org/doi/pdf/10.1145/3560815).

   
1. **How Does In-Context Learning Help Prompt Tuning?** *Simeng Sun, Yang Liu, Dan Iter, Chenguang Zhu, and Mohit Iyyer.* <ins>Preprint</ins> 2023. [[pdf](https://arxiv.org/pdf/2302.11521.pdf)]; [[code](link)]. 
   
2. **Demystifying Prompts in Language Models via Perplexity Estimation.** *Hila Gonen, Srini Iyer, Terra Blevins, Noah A. Smith, and Luke Zettlemoyer.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2212.04037.pdf)]; [[code](link)]. 
   
3. **RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning.** *Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, and et al.* <ins>EMNLP</ins> 2022. [[pdf](https://arxiv.org/pdf/2205.12548.pdf)]; [[code](https://github.com/mingkaid/rl-prompt)]. 
   
4. **PPT: Pre-trained Prompt Tuning for Few-shot Learning.** *Yuxian Gu, Xu Han, Zhiyuan Liu, and Minlie Huang.* <ins>ACL</ins> 2022. [[pdf](https://arxiv.org/pdf/2109.04332.pdf)]; [[code](link)]. 
   
5. **KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction.** *Xiang Chen, Ningyu Zhang, Xin Xie, and et al.* <ins>WWW</ins> 2022. [[pdf](http://128.84.21.203/pdf/2104.07650)]; [[code](link)].
    
6. **GPT Understands, Too.** *Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang.* <ins>Preprint</ins> 2021. [[pdf](https://arxiv.org/pdf/2103.10385.pdf)]; [[code](link)].
   
7.  **Few-Shot Text Generation with Natural Language Instructions.** *Timo Schick and Hinrich Schütze.* <ins>EMNLP</ins> 2021. [[pdf](https://aclanthology.org/2021.emnlp-main.32.pdf)]; [[code](link)]. 
   
8.  **It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners.** *Timo Schick and Hinrich Schütze.* <ins>NAACL</ins> 2021. [[pdf](https://aclanthology.org/2021.naacl-main.185.pdf)]; [[code]()]. 
   
9.  **Learning How to Ask: Querying LMs with Mixtures of Soft Prompts.** *Guanghui Qin and Jason Eisner.* <ins>NAACL</ins> 2021. [[pdf](https://aclanthology.org/2021.naacl-main.410.pdf)]; [[code](link)]. 
   
10. **Prefix-Tuning: Optimizing Continuous Prompts for Generation.** *Xiang Lisa Li and Percy Liang.* <ins>ACL</ins> 2021. [[pdf](https://aclanthology.org/2021.acl-long.353.pdf)]; [[code](link)]. 
   
11. **Making Pre-trained Language Models Better Few-shot Learners.** *Tianyu Gao, Adam Fisch, and Danqi Chen.* <ins>ACL</ins> 2021. [[pdf](https://aclanthology.org/2021.acl-long.295.pdf)]; [[code]()]. 
   
12. **Template-Based Named Entity Recognition Using BART.** *Leyang Cui, Yu Wu, Jian Liu, Sen Yang, and Yue Zhang.* <ins>Findings of ACL</ins> 2021. [[pdf](https://aclanthology.org/2021.findings-acl.161.pdf)]; [[code](link)]. 
   
13. **Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference.** *Timo Schick and Hinrich Schütze.* <ins>EACL</ins> 2021. [[pdf](https://aclanthology.org/2021.eacl-main.20.pdf)]; [[code](link)].
   
14. **Language Models are Unsupervised Multitask Learners.** *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.* <ins>Preprint</ins> 2019. [[pdf](https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf)]. 
   

### 2.3 Human-oriented Instruction

1. **Paper Title.** *Author 1, Author 2, and Author 3.* <ins>Conference/Journal/Preprint</ins> Year. [[pdf](link)]; [[other resources](link)].
   
2. **InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis.** *Kevin Scaria, Himanshu Gupta, Saurabh Arjun Sawant, Swaroop Mishra, and Chitta Baral.* <ins>Preprint</ins> 2023. [[pdf](https://arxiv.org/pdf/2302.08624.pdf)]; [[other resources](link)].
   
3. **HINT: Hypernetwork Instruction Tuning for Efficient Zero-Shot Generalisation.** *Hamish Ivison, Akshita Bhagia, Yizhong Wang, Hannaneh Hajishirzi, and Matthew Peters.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2212.10315.pdf)]; [[code](link)].

4. **Boosting Natural Language Generation from Instructions with Meta-Learning.** *Budhaditya Deb, Guoqing Zheng, and Ahmed Hassan Awadallah.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2210.11617.pdf)]. 
   
5. **GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models.** *Archiki Prasad, Peter Hase, Xiang Zhou, and Mohit Bansal.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2203.07281.pdf)]; [[code](link)].
   
6. **ConTinTin: Continual Learning from Task Instructions.** *Wenpeng Yin, Jia Li, and Caiming Xiong.* <ins>ACL</ins> 2022. [[pdf](https://aclanthology.org/2022.acl-long.218.pdf)]; [[code]()]. 
   
7. **InstructDial: Improving Zero and Few-shot Generalization in Dialogue through Instruction Tuning.** *Prakhar Gupta, Cathy Jiao, Yi-Ting Yeh, Shikib Mehri, Maxine Eskenazi, and Jeffrey P. Bigham.* <ins>EMNLP</ins> 2022. [[pdf]([link](http://128.84.21.203/pdf/2205.12673))]; [[code](link)]. 
   
8. **Learning to Generate Task-Specific Adapters from Task Description.** *Qinyuan Ye and Xiang Ren.* <ins>ACL</ins> 2021. [[pdf](https://aclanthology.org/2021.acl-short.82.pdf)]; [[code]()]. <!-- TODO -->
   
9.  **The Turking Test: Can Language Models Understand Instructions?** *Avia Efrat and Omer Levy.* <ins>Preprint</ins> 2020. [[pdf](https://arxiv.org/pdf/2010.11982.pdf)]; [[code]()]. 


## 3. 📊 Analyses

### 3.1 Scale
The model and task scale are found to be important for instruction-based fine-tuning. Basically, the larger model scale brings more benefits to the generalization. So as the task scale, however, some works also raised objections (e.g., [Jang et al.](https://arxiv.org/pdf/2302.03202.pdf)).

1. **Paper Title.** *Author 1, Author 2, and Author 3.* <ins>Conference/Journal/Preprint</ins> Year. [[pdf](link)]; [[other resources](link)]. 
   
2. **Exploring the Benefits of Training Expert Language Models over Instruction Tuning.** *Joel Jang, Seungone Kim, Seonghyeon Ye, and et al.* <ins>Preprint</ins> 2023. [[pdf](https://arxiv.org/pdf/2302.03202.pdf)]; [[code](https://github.com/joeljang/ELM)]. 
    
3. **UL2: Unifying Language Learning Paradigms.** *Yi Tay, Mostafa Dehghani, Vinh Q. Tran, and et al.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2205.05131.pdf)]; [[checkpoint](https://huggingface.co/google/flan-ul2)].  
   
4. **The Flan Collection: Designing Data and Methods for Effective Instruction Tuning.** *Shayne Longpre, Le Hou, Tu Vu, and et al.* <ins>Preprint</ins> 2023. [[pdf](https://arxiv.org/pdf/2301.13688.pdf)]; [[code](https://github.com/google-research/FLAN/tree/main/flan/v2)]; [[corpus](https://huggingface.co/datasets/SirNeural/flan_v2)].  
   
5. **OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization.** *Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, and et al.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2212.12017.pdf)].   
   
6. **Scaling Instruction-Finetuned Language Models.** *Hyung Won Chung, Le Hou, Shayne Longpre, and et al.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2210.11416.pdf)]; [[checkpoint](https://huggingface.co/docs/transformers/model_doc/flan-t5)].  
   
7. **Multitask Prompted Training Enables Zero-Shot Task Generalization.** *Victor Sanh, Albert Webson, Colin Raffel, and et al.* <ins>ICLR</ins> 2022. [[pdf](https://openreview.net/pdf?id=9Vrb9D0WI4)]; [[code]()]. 
8.  **Finetuned Language Models are Zero-Shot Learners.** *Jason Wei, Maarten Bosma, Vincent Zhao, and et al.* <ins>ICLR</ins> 2022. [[pdf](https://openreview.net/pdf?id=gEZrGCozdqR)]; [[code](link)].
9.  **ZeroPrompt: Scaling Prompt-Based Pretraining to 1,000 Tasks Improves Zero-Shot Generalization.** *Hanwei Xu, Yujun Chen, Yulun Du, Nan Shao, Yanggang Wang, Haiyu Li, and Zhilin Yang.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2201.06910.pdf)]; [[other resources](link)]. 
10. **The Power of Scale for Parameter-Efficient Prompt Tuning.** *Brian Lester, Rami Al-Rfou, and Noah Constant.* <ins>EMNLP</ins> 2021. [[pdf](https://aclanthology.org/2021.emnlp-main.243.pdf)]; [[code](link)]. 

### 3.2 Explanability

1. **Paper Title.** *Author 1, Author 2, and Author 3.* <ins>Conference/Journal/Preprint</ins> Year. [[pdf](link)]; [[other resources](link)].
2. **Prompt Waywardness: The Curious Case of Discretized Interpretation of Continuous Prompts.** *Daniel Khashabi, Xinxi Lyu, Sewon Min, and et al.* <ins>NAACL</ins> 2022. [[pdf](https://aclanthology.org/2022.naacl-main.266.pdf)]; [[code](link)]. 
3. **Do Prompt-Based Models Really Understand the Meaning of Their Prompts?.** *Albert Webson and Ellie Pavlick.* <ins>NAACL</ins> 2022. [[pdf](https://aclanthology.org/2022.naacl-main.167.pdf)]; [[code](https://github.com/awebson/prompt_semantics)].
4. **Reframing Instructional Prompts to GPTk’s Language.** *Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, and Hannaneh Hajishirzi.* <ins>Findings of ACL</ins> 2022. [[pdf](https://aclanthology.org/2022.findings-acl.50.pdf)]; [[code](https://github.com/allenai/reframing/)]. 


### 3.3 Robustness

1. **Paper Title.** *Author 1, Author 2, and Author 3.* <ins>Conference/Journal/Preprint</ins> Year. [[pdf](link)]; [[other resources](link)].
   
2. **Robustness of Learning from Task Instructions.** *Jiasheng Gu, Hanzi Xu, Liangyu Nie, and Wenpeng Yin.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2212.03813.pdf)]; [[code](link)]. 

3. **Learning from Task Descriptions.** *Orion Weller, Nicholas Lourie, Matt Gardner, and Matthew E. Peters.* <ins>EMNLP</ins> 2020. [[pdf](https://aclanthology.org/2020.emnlp-main.105.pdf)]; [[code](https://github.com/allenai/zest)]; [[dataset](https://allenai.org/data/zest)]. 

### 3.4 Negation

### 3.5 Others

## 4. 🤖 Applications

### 4.1 Human-Computer Interaction

### 4.2 Data and Feature Augmentation

### 4.3 General-purpose Language Models

## 5. 📚 Corpora

1. **Paper Title.** *Author 1, Author 2, and Author 3.* <ins>Conference/Journal/Preprint</ins> Year. [[pdf](link)]; [[other resources](link)].
   
2. **Self-Instruct: Aligning Language Model with Self Generated Instructions.** *Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2212.10560.pdf)]; [[corpus](https://github.com/yizhongw/self-instruct)].
   
3. **Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor.** *Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2212.09689.pdf)]; [[corpus](https://github.com/orhonovich/unnatural-instructions)]. 
   
4. **Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks.** *Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, and et al.* <ins>EMNLP</ins> 2022. [[pdf](https://arxiv.org/pdf/2204.07705.pdf)]; [[corpus](https://instructions.apps.allenai.org/)]. 
   
5. **Cross-Task Generalization via Natural Language Crowdsourcing Instructions.** *Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi.* <ins>ACL</ins> 2022. [[pdf](https://aclanthology.org/2022.acl-long.244.pdf)]; [[corpus](https://instructions.apps.allenai.org/)]. 
   
6. **PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts.** *Stephen Bach, Victor Sanh, Zheng Xin Yong, and et al.* <ins>ACL</ins> 2022. [[pdf](https://aclanthology.org/2022.acl-demo.9.pdf)]; [[toolkit](https://github.com/bigscience-workshop/promptsource)]; [[corpus](https://huggingface.co/datasets/bigscience/P3)].

## 6. 🗒️ Other Papers

Including instruction induction papers.

1. **Paper Title.** *Author 1, Author 2, and Author 3.* <ins>Conference/Journal/Preprint</ins> Year. [[pdf](link)]; [[other resources](link)].
   
2. **Instruction Induction: From Few Examples to Natural Language Task Descriptions.** *Or Honovich, Uri Shaham, Samuel R. Bowman, and Omer Levy.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2205.10782.pdf)]; [[code](https://github.com/orhonovich/instruction-induction)].

Human feedback vs model feedback

1. **Paper Title.** *Author 1, Author 2, and Author 3.* <ins>Conference/Journal/Preprint</ins> Year. [[pdf](link)]; [[other resources](link)].
   
2. **Constitutional AI: Harmlessness from AI Feedback.** *Yuntao Bai, Saurav Kadavath, Sandipan Kundu, and et al.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2212.08073.pdf)]; [[corpus](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)].
   
3. **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.** *Yuntao Bai, Andy Jones, Kamal Ndousse, and et al.* <ins>Preprint</ins> 2022. [[pdf](https://arxiv.org/pdf/2204.05862.pdf)]; [[corpus](https://github.com/anthropics/hh-rlhf)]. 

ChatGPT related

Scalable oversight

---

<!-- TODO: tweets & slides? -->