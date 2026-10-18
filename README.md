# HW4P1 and HW4P2

## HW4P2 WebNLG SFT

Continue in your P1 folder and open `HW4P2_Student_Notebook.ipynb`.
Both notebooks use the same `hw4lib/`. Complete the added P2 dataset,
SFT trainer and LoRA TODOs. Keep your P1 implementations and checkpoint.

See [P2 setup](HW4P2_README.md) for adding the files and dependencies.

## HW4P1 Decoder-Only Language Modeling

Open `HW4P1_Student_Notebook.ipynb` from this folder. Follow the setup section for
Local, Colab, Kaggle, or PSC, and use the `hw4` environment for local work.
Colab uses `requirements-colab.txt` to keep its installed PyTorch/CUDA stack.
Use the notebook's manual data path if the Kaggle download fails.

- `mytorch/nn/`: copy your HW3 prerequisites and complete GQA/MQA.
- `hw4lib/`: complete the marked dataset, model, decoding, and trainer TODOs.
- `tests/`: run the correctness tests beneath each implementation section in the notebook.

The small models used by the tests are not your training configuration. After
the implementation tests pass, run the mini experiment, choose your final
configuration, and complete full training and evaluation.

The notebook creates `handin.tar.gz` containing your code and training evidence.
Submit that file to Gradescope. Checkpoint weights are not included.
Do not save API keys in the notebook or source files.
