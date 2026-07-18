"""
Prompt templates for MultiTask-CodePilot.

Converts a unified dataset record into an instruction prompt
that will be used for Qwen fine-tuning.
"""


SYSTEM_PROMPT = (
    "You are an expert software engineering assistant. "
    "Follow the user's instruction carefully and provide accurate responses."
)


def build_prompt(record):
    """
    Convert a dataset record into a prompt.

    Args:
        record (dict)

    Returns:
        str
    """

    task = record["task"]

    # ---------------------------------------------------------
    # Code Generation
    # ---------------------------------------------------------

    if task == "code_generation":

        instruction = (
            "Write Python code to solve the following programming problem."
        )

    # ---------------------------------------------------------
    # Documentation
    # ---------------------------------------------------------

    elif task == "documentation":

        instruction = (
            "Generate clear and concise documentation for the following code."
        )

    # ---------------------------------------------------------
    # Translation
    # ---------------------------------------------------------

    elif task == "translation":

        src = record["parameters"]["source_language"]
        tgt = record["parameters"]["target_language"]

        instruction = (
            f"Translate the following {src} code into {tgt}."
        )

    else:
        raise ValueError(f"Unknown task: {task}")

    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>

<|im_start|>user
### Task
{task}

### Instruction
{instruction}

### Input
{record["input"]}
<|im_end|>

<|im_start|>assistant
{record["output"]}
<|im_end|>
"""

    return prompt