"""
Adaptive-RAG Classifier Loader
Specialized for Adaptive-RAG with only Z/S/M actions.
"""

import os
import re
import torch

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class AdaptiveClassifierLoader:
    """Load and run inference with Adaptive-RAG classifier (Z/S/M only)."""

    def __init__(self, config):
        self.config = config
        self.base_model_path = config['classifier']['base_model_path']
        self.lora_path = config['classifier']['lora_adapter_path']
        self.max_seq_length = config['classifier']['max_seq_length']
        self.device = config['classifier']['device']
        self.model = None
        self.tokenizer = None

        # Adaptive-RAG specific system prompt (must match training prompt in finetune.py)
        self.system_prompt = """You are an expert RAG router. Analyze the user query complexity and determine the optimal retrieval strategy.
Output the analysis in the following format:
Analysis: <reasoning process>
Complexity: <L0/L1/L2>
Index: <None/BM25>
Action: <Z/S/M>"""

    def load_model(self):
        """Load base model + LoRA adapter using unsloth."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "unsloth is required for classifier loading. "
                "Install with: pip install unsloth"
            )

        print(f"Loading base model from {self.base_model_path}...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,
            local_files_only=True,
        )

        print(f"Loading LoRA adapter from {self.lora_path}...")

        model.load_adapter(self.lora_path)
        FastLanguageModel.for_inference(model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.model = model
        self.tokenizer = tokenizer

        print("Adaptive-RAG classifier loaded successfully")

        return model, tokenizer

    def classify_question(self, question_text: str):
        """Run inference on a single question."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question_text}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config['classifier']['max_new_tokens'],
                temperature=self.config['classifier']['temperature'],
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        action = self.parse_action(response)

        return {
            'action': action,
            'full_response': response
        }

    def parse_action(self, text: str) -> str:
        """
        Parse action from model output.
        Only accepts Z, S, M for Adaptive-RAG.
        """
        # Valid actions for Adaptive-RAG
        valid_actions = ["Z", "S", "M"]

        # Try standard regex: "Action: <value>"
        match = re.search(r"Action:\s*([A-Za-z0-9\-]+)", text)
        if match:
            action = match.group(1).strip().rstrip('.')

            # Direct match
            if action in valid_actions:
                return action

            # Map variants to base actions
            if action.startswith("S-") or action == "S":
                return "S"
            if action.startswith("Z"):
                return "Z"
            if action.startswith("M"):
                return "M"

        # Fallback: search in last line
        lines = text.strip().split('\n')
        if lines:
            last_line = lines[-1]
            # Check for M first (to avoid matching M in other words)
            if re.search(r'\bM\b', last_line):
                return "M"
            if re.search(r'\bS\b', last_line) or "S-" in last_line:
                return "S"
            if re.search(r'\bZ\b', last_line):
                return "Z"

        # Search entire text
        for valid in valid_actions:
            if re.search(rf'\b{valid}\b', text):
                return valid

        # Default fallback: use S (single retrieval) as safe default
        return "S"

    def classify_batch(self, questions: list):
        """Classify multiple questions using GPU batch inference."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not questions:
            return []

        # Batch construct messages
        all_messages = []
        for question in questions:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question}
            ]
            all_messages.append(messages)

        # Batch tokenize
        input_ids_list = []
        for messages in all_messages:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            input_ids_list.append(input_ids[0])

        # Left padding
        max_length = max(len(ids) for ids in input_ids_list)
        padded_inputs = []
        for ids in input_ids_list:
            padding_length = max_length - len(ids)
            if padding_length > 0:
                padded = torch.cat([
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=ids.dtype),
                    ids
                ])
            else:
                padded = ids
            padded_inputs.append(padded)

        padded_inputs = torch.stack(padded_inputs)
        attention_mask = (padded_inputs != self.tokenizer.pad_token_id).long()
        input_lengths = [len(ids) for ids in input_ids_list]

        # Batch generation
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

        # Batch decode
        results = []
        for i in range(len(questions)):
            generated_tokens = outputs[i][input_lengths[i]:]
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            action = self.parse_action(response)

            results.append({
                'action': action,
                'full_response': response
            })

        return results
