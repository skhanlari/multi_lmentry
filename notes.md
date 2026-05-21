- Persian Dataset
    - Lexicon
    - Homophone Pairs
    - 1000 Sentences
    - Lmentry_Prompt_Translation
    - Scorers
_________________________________________________________________________________________

- Dataset Compresion (19 different Ratios)
    - Without_Template_based_Compresion (Spearman & Pearson = 0.9)
        - Random
        - Clustering (similarity_threshold = 0.9)
        - Difficulty (Easy, Mid, Hard, Stratified)

    - Template_based_Compresion (Spearman & Pearson = 0.9, 0.95, 0.99)
        - Random
        - Clustering (similarity_threshold = 0.9, 0.95, 0.99)
        - Stratified_Difficulty

    - Cross_Testing
        - Template_based
        - Train --> 18 Models --> Spearman, Pearson, MSE
        - Test --> 1 Model --> MSE

    - Plots ??
_________________________________________________________________________________________

- LLM_Evaluation

    - JUDGE_MODELS
        - "Qwen/Qwen2.5-7B-Instruct",
        - "mistralai/Mistral-7B-Instruct-v0.3",
        - "microsoft/Phi-3-mini-4k-instruct",
        - "meta-llama/Llama-3.1-8B-Instruct",

    - FOR aLL Languages and Tasks
        - Use all JUDGE_MODELS
        - Also use GOLD (if exist)

    - Output

        - evaluated_outputs/
            - llm_judge / judge_model / language / task / source_model.json
            - gold_eval / language / task / source_model.json

        - Content of one judge file:

            {
            "1": {
                "language": "en",
                "task": "rhyming_word",
                "source_model": "ALIA-40b",
                "judge_model": "Qwen/Qwen2.5-7B-Instruct",
                "input": "Write a word that rhymes with 'cat'",
                "prediction": "bat",
                "judge_label": 1
            }
            }

        - Content of a gold file:

            {
            "1": {
                "language": "en",
                "task": "rhyming_word",
                "source_model": "ALIA-40b",
                "input": "Write a word that rhymes with 'cat'",
                "prediction": "bat",
                "gold_label": 1,
                "predicted_label": 1,
                "correct": 1
            }
            }
