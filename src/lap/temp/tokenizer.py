import numpy as np

class PaligemmaTokeniazer:
    def __init__(self, vocab):
        self.vocab = vocab

    def tokenize(
            self,
            prompt:str,
            reasoning:str | None = None,
            state: np.ndarray | None = None,
            state_type:str | None = None,
            langact:str | None = None,
            ):
        
        """
        if langact is None (legacy)
        """
        formatted_prompt = fmt.format_prompt(
            prompt,
            state,
            state_type,
        )

        tokens = self._tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)

        add_eos_for_reasoning = langact is None

        clean_reason = reasoning.strip().replace("_", " ").replace("\n", " ")

        tokens += self._tokenizer.encode(clean_reason, add_bos=False, add_eos = add_eos_for_reasoning)
        reasoning_end = len(tokens)

        langact_start = reasoning_end
        langact_end = reasoning_end

        if langact is not None:
            sep_tokens = self._tokenizer.encode("<|action|>", add_bos=False, add_eos=False)
            tokens += sep_tokens
            langact_start = len(tokens)
            tokens += self._tokenizer.encode(langact, add_bos=False, add_eos=True)