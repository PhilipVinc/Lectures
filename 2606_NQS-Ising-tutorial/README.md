# NQS tutorial: the 2D Ising transition

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhilipVinc/Lectures/blob/main/2606_NQS-Ising-tutorial/ising_nqs_exercise.ipynb)

A hands-on tutorial using Neural Quantum States (NQS) with
[NetKet](https://www.netket.org/) to study the ground state of the 2D
transverse-field Ising model across its quantum phase transition
($h_c/J \approx 3.044$).

You need both files!!

## Running it

To run on colab, click the badge above and run the first cell — it installs the dependencies and downloads `ansatze.py` for you.

### Locally

Requires `netket>=3.22.2`, `nqxpack` and `matplotlib`. Either

```bash
pip install netket>=3.22.2 nqxpack matplotlib
```

or, from this directory,

```bash
pip install .
```

then launch Jupyter **from this directory** (so that `import ansatze` works) and
open the notebook. A laptop CPU is enough; a GPU speeds things up noticeably (the
CNN in particular is slow on CPU).
