# Dataset Selection

## Objective

Train a **single LoRA fine-tuned model** to perform three software engineering tasks:

* Natural Language → Python Code
* Python → Java Translation
* Code → Documentation

---

## Dataset Selection

| Capability       | Training Dataset | Evaluation          |
| ---------------- | ---------------- | ------------------- |
| Code Generation  | MBPP             | HumanEval+          |
| Code Translation | XLCoST           | XLCoST Test Split   |
| Documentation    | DocuMint         | DocuMint Test Split |

---

## Why these datasets?

### MBPP

* High-quality human-written programming tasks
* Natural Language → Python code generation
* Standard benchmark for code generation

### XLCoST

* Parallel source code in multiple languages
* Ideal for Python ↔ Java translation
* Easy to preprocess

### DocuMint

* Purpose-built for documentation generation
* Large, curated dataset
* Better suited for fine-tuning than generic code datasets

---

## Training Strategy

* Convert all datasets into a unified format.
* Fine-tune **one model** instead of three separate models.
* Select task during inference using a task-specific prompt.

---

## Dataset schema
### PL to PL
{
    "task": "translation",

    "parameters": {
        "source_language": "python",
        "target_language": "java"
    },

    "input": "...",

    "output": "..."
}

### NL to PL
{
    "task": "code_generation",

    "parameters": {
        "language": "python"
    },

    "input": "Write a function to calculate factorial.",

    "output": "def factorial(n): ..."
}

### Doc Generation

{
    "task": "documentation",

    "parameters": {
        "language": "python"
    },

    "input": "def add(a,b): return a+b",

    "output": "Adds two numbers."
}



---

## Future Scope

The same approach can be extended to additional capabilities such as:

* Code Explanation
* Unit Test Generation
* Commit Message Generation
* Code Review

