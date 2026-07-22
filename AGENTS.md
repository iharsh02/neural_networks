# AGENTS.md

## Project purpose

This is a personal, code-along learning repository for developing a deep understanding of neural networks, from first principles through complete language models.

The curriculum follows two main resources:

- Andrej Karpathy's *Neural Networks: Zero to Hero* series, progressing from derivatives and manual backpropagation through `micrograd`-style neural-network building blocks, `makemore`, WaveNet, and a GPT-style model.
- Daniel Bourke's PyTorch fundamentals series, covering tensors, a practical training workflow, and the core PyTorch tools used in modern deep learning.

The user watches the videos, writes code alongside them, and records their own explanations in notebooks. Treat their code and notes as the primary learning artifact. The agent's role is to turn each notebook into a clearer, more complete reference without taking ownership away from the learner.

## Repository map

- `notebook/pytorch-fundamentals/` — PyTorch tensors and end-to-end training workflow.
- `notebook/backpropgation/` — derivatives, manual backpropagation, and a neural-network library built from scratch.
- `notebook/backpropninja/` — manual backpropagation exercises for the MLP language model.
- `notebook/bigram-make-more/` — character-level bigram language model.
- `notebook/bigram-make-more-mlp/` — MLP character-level language model.
- `notebook/wavenet/` — WaveNet-style character-level model.
- `notebook/nano-gpt/` — GPT-style transformer experiment and its local documentation.
- `notebook/gpt-tokenizer/` — GPT-style tokenizer: byte-pair encoding (BPE) and SentencePiece.
- `models/` — saved model weights.

Keep existing directory and file names unchanged, including historical spellings such as `backpropgation`.

## Working with notebooks

When asked to help with a notebook, improve it as a learning reference while preserving the user's work.

### Do

- Expand existing comments and markdown with fuller, more precise explanations. Preserve the user's voice: build on what they wrote instead of replacing it.
- Add missing context: why a design decision was made, relevant historical background, and links to earlier or later concepts in the curriculum.
- Improve technical precision gently. Correct vague or inaccurate explanations without becoming pedantic.
- Add focused reinforcing material when it clarifies the lesson: a small worked example, an explanatory markdown cell, or a lightweight Matplotlib/Seaborn visualization.
- Connect the current topic to the wider ML path. For example, relate scalar autodiff to PyTorch autograd, manual backpropagation to training an MLP, or the bigram model to later language models.
- Keep executable examples small, deterministic where practical, and compatible with the notebook's existing imports and runtime.
- Before changing executable logic, inspect surrounding cells and notebook execution order. A notebook may fail because its kernel was restarted or prerequisite cells were run out of order rather than because a helper is missing.

### Do not

- Never remove, replace, or refactor away the user's existing code. Additions and narrowly scoped comment corrections are allowed; preservation is mandatory.
- Do not rewrite the user's explanations in a different voice. Expand and clarify them instead.
- Do not add material far beyond the video or notebook's current topic. Be thorough but relevant and illustrative rather than exhaustive.
- Do not put broad contextual teaching in code comments. Use a markdown cell above or below the code for conceptual discussion, trade-offs, history, comparisons, or curriculum connections. Code comments should state what the code does or why that specific implementation step is necessary.
- Do not assume a missing-name error means source code is missing. Check cell definitions and execution state first.

## Notebook safety rules

These rules are mandatory.

1. Before editing a notebook, ask the user to save any unsaved IDE changes. Notebook file edits can otherwise overwrite work that is visible in the editor but not yet saved.
2. Never run a tool or script that rewrites an entire `.ipynb` file. Make targeted, cell-level edits only.
3. Never delete a user cell or user code. When a correction is needed, preserve the original material and make the smallest safe addition or comment update.
4. Verify the placement of every inserted cell. If using a cell editor whose insert operation places a cell *after* the selected cell, select the preceding cell when the new markdown must appear before a code cell.
5. After edits, re-check cell order and only run the smallest relevant sequence of cells unless the user asks to run the full notebook.
6. If the kernel state is unclear, explain the required prerequisite cells or recommend **Run All Cells** rather than duplicating definitions.

## Python environment and validation

- Read `README.md` and `requirements.txt` before assuming dependencies or commands.
- The standard local environment is a virtual environment named `.venv`:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  jupyter notebook
  ```

- PyTorch installation is intentionally left to the platform-appropriate command from the official PyTorch selector; do not blindly add a generic PyTorch install command without confirming the target platform/CUDA needs.
- Prefer project-local paths and notebook-relative data access over machine-specific absolute paths.
- Validate changes proportionally: check syntax/imports and the directly affected cells or script. Do not claim a notebook is fully runnable unless its relevant cells have actually been executed in a compatible environment.

## Communication expectations

- Lead with what changed or what the diagnosis is, then give only the technical details needed to understand it.
- Explain completed work in learning terms: what concept is now clearer, how it fits the progression, and any prerequisite execution order the user should know.
- Flag assumptions and blockers plainly. Ask for direction before making a change that would alter the scope or learning approach rather than merely improve clarity.
