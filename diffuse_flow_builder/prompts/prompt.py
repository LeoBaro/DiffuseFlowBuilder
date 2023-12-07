from dataclasses import dataclass   

@dataclass
class Prompt:
    prefix:      str | None = None
    subject:     str | None = None
    enhancement: str | None = None

    def get_str_prompt(self):
        return f"{self.prefix} {self.subject}, {self.enhancement}"
    
    @staticmethod
    def from_dict(d):
        return Prompt(d["prompt_prefix"], d["prompt_subject"], d["prompt_enanchment"])
    
    def combine_with(self, prompt):

        combined_prompt = Prompt()

        if self.prefix is not None:
            combined_prompt.prefix = self.prefix
        else:
            combined_prompt.prefix = prompt.prefix

        if self.subject is not None:
            combined_prompt.subject = self.subject
        else:
            combined_prompt.subject = prompt.subject

        if self.enhancement is not None:
            combined_prompt.enhancement = self.enhancement
        else:
            combined_prompt.enhancement = prompt.enhancement

        return combined_prompt