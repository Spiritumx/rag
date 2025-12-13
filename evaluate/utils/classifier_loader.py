"""
Classifier model loader for Qwen 2.5-3B with LoRA adapters.
Based on classifier/train/evaluate.py
"""

import os
import re
import torch

# Force offline mode (same as classifier/train/evaluate.py)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class ClassifierLoader:
    """Load and run inference with Qwen 2.5-3B LoRA classifier."""

    def __init__(self, config):
        self.config = config
        self.base_model_path = config['classifier']['base_model_path']
        self.lora_path = config['classifier']['lora_adapter_path']
        self.max_seq_length = config['classifier']['max_seq_length']
        self.device = config['classifier']['device']
        self.model = None
        self.tokenizer = None

        # System prompt for classification
        self.system_prompt = """You are an expert RAG router. Analyze the user query complexity and determine the optimal retrieval strategy.
Output the analysis in the following format:
Analysis: <reasoning process>
Complexity: <L0/L1/L2>
Index: <None/Lexical/Semantic/Hybrid>
Action: <Z/S-Sparse/S-Dense/S-Hybrid/M>"""

    def load_model(self):
        """
        Load base model + LoRA adapter using unsloth.
        Returns (model, tokenizer)
        """
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "unsloth is required for classifier loading. "
                "Install with: pip install unsloth"
            )

        print(f"Loading base model from {self.base_model_path}...")

        # Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,
            local_files_only=True,
        )

        print(f"Loading LoRA adapter from {self.lora_path}...")

        # Load LoRA adapter
        model.load_adapter(self.lora_path)

        # Set to inference mode
        FastLanguageModel.for_inference(model)

        # Fix tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.model = model
        self.tokenizer = tokenizer

        print("✓ Classifier model loaded successfully")

        return model, tokenizer

    def classify_question(self, question_text: str):
        """
        Run inference on a single question.

        Args:
            question_text: The question to classify

        Returns:
            dict: Classification results with keys:
                - action: The predicted action (Z/S-Sparse/S-Dense/S-Hybrid/M)
                - full_response: The full model response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Format messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question_text}
        ]

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config['classifier']['max_new_tokens'],
                temperature=self.config['classifier']['temperature'],
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        # Parse action
        action = self.parse_action(response)

        return {
            'action': action,
            'full_response': response
        }

    def parse_action(self, text: str) -> str:
        """
        Parse action from model output.
        Based on classifier/train/evaluate.py lines 35-59

        Args:
            text: Model output text

        Returns:
            Predicted action label
        """
        valid_actions = ["Z", "S-Sparse", "S-Dense", "S-Hybrid", "M"]

        # Try standard regex: "Action: <value>"
        match = re.search(r"Action:\s*([A-Za-z0-9\-]+)", text)
        if match:
            action = match.group(1).strip().rstrip('.')
            if action in valid_actions:
                return action

        # Fallback: search in last line for valid actions
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1]
            for valid in valid_actions:
                if valid in last_line:
                    return valid

        # If still not found, search entire text
        for valid in valid_actions:
            if valid in text:
                return valid

        return "Unknown"

    def classify_batch(self, questions: list):
        """
        Classify multiple questions using GPU batch inference.

        Args:
            questions: List of question texts

        Returns:
            List of classification results
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not questions:
            return []

        # 1. Batch construct messages
        all_messages = []
        for question in questions:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ]
            all_messages.append(messages)

        # 2. Batch tokenize (apply_chat_template individually, then combine)
        input_ids_list = []
        for messages in all_messages:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            input_ids_list.append(input_ids[0])  # Extract [seq_len]

        # 3. Padding to unified length (left padding for decoder-only models)
        # Manual left padding (pad_sequence does right padding by default)
        max_length = max(len(ids) for ids in input_ids_list)
        padded_inputs = []
        for ids in input_ids_list:
            padding_length = max_length - len(ids)
            # Left padding: add padding tokens at the beginning
            if padding_length > 0:
                padded = torch.cat([
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype),
                    ids
                ])
            else:
                padded = ids
            padded_inputs.append(padded)

        padded_inputs = torch.stack(padded_inputs)

        # 4. Create attention_mask (critical!)
        attention_mask = (padded_inputs != self.tokenizer.pad_token_id).long()

        # 5. Record input length for each sample (for decoding)
        input_lengths = [len(ids) for ids in input_ids_list]

        # 6. Batch generation
        with torch.no_grad():
            outputs = self.model.generate(
                padded_inputs.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=self.config['classifier']['max_new_tokens'],
                temperature=self.config['classifier']['temperature'],
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 7. Batch decode (only the generated part)
        results = []
        for i in range(len(questions)):
            # Extract only the generated tokens
            generated_tokens = outputs[i][input_lengths[i]:]
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            # Parse action
            action = self.parse_action(response)

            results.append({
                'action': action,
                'full_response': response
            })

        return results
