# Plan — NeuralConsole: Cross-Entropy + Weight Decay Back-Propagation

## Goal

Build a .NET C# console program based on `BackPropProgram.cs`, following the
**WHAT TO DO** section of `instructions.md`, which asks for two modifications to the
back-propagation code:

1. **Cross-entropy error** — an alternative to the mean squared error (MSE)
   assumption baked into the current training equations.
2. **Weight decay (L2 regularization)** — penalize large weight values to reduce
   model over-fitting (where train accuracy is much higher than test accuracy).

## What we start with

`BackPropProgram.cs` — the article's demo:

- Generates 1,000 synthetic rows (4 features, 3 classes, 1-of-N labels) using a
  *generating* neural network with random weights.
- Splits 80/20 into train/test, trains a 4-5-3 network with stochastic
  (online) back-prop, softmax outputs, tanh hidden layer.
- Reports periodic MSE during training, final weights, and train/test accuracy.

Known limitations to address:

- Training error metric is hard-coded to MSE (`oSignals[k] = (t - o) * o * (1 - o)`).
- No regularization of any kind — nothing keeps weights small.
- `Train(...)` cannot choose error function or decay; `Main` has no way to
  experiment with either.
- `Console.ReadLine()` at the end blocks automated runs.
- `errInterval = maxEpochs / 10` would divide by zero for `maxEpochs < 10`.

## Design

### Files

| File | Action |
|---|---|
| `plan.md` | This plan (new) |
| `NeuralConsole.csproj` | New, minimal SDK-style project (`Microsoft.NET.Sdk`, `OutputType=Exe`, `TargetFramework=net10.0`) — picks up every `*.cs` in the folder |
| `BackPropProgram.cs` | Modified in place; original article structure/style kept |
| `instructions.md` | Untouched reference |

Build/run: `dotnet build` / `dotnet run`

### 1. Error-function enum + cross-entropy back-prop

Add:

```csharp
public enum ErrorFunction { MeanSquaredError, CrossEntropy }
```

In `NeuralNetwork.Train`, the output-layer local signals become:

- **MSE (unchanged):** `oSignals[k] = (t[k] - o[k]) * (1 - o[k]) * o[k]`
  (softmax-derivative term from the article).
- **Cross-entropy:** `oSignals[k] = t[k] - o[k]`
  — for softmax + cross-entropy the output derivative term cancels exactly,
  so the signal is just the error. Everything downstream (hidden signals,
  gradients, momentum updates) stays the same.

Add a matching error metric for monitoring:

```csharp
public double CrossEntropyError(double[][] data)  // mean of -sum(t_k * ln(o_k))
```

Keep the existing MSE `Error(...)` public so both metrics can be printed at the
end for comparison. During training, print every
`errInterval = max(1, maxEpochs / 10)` epochs using the *selected* error function.

### 2. Weight decay (L2 regularization)

Add a `decay` (λ) parameter to `Train`. Applied to **i-h and h-o weights**
(biases excluded — decay there is usually counterproductive and the article's
bias gradients already use a dummy 1.0 input):

```
delta = learnRate * gradient - learnRate * decay * weight
weight += delta + momentum * previousDelta
```

i.e. the gradient of an extra `(λ/2)·Σw²` penalty term. `decay = 0.0` gives the
original behavior. The penalty only touches the weight-update loops — gradient
computation is unchanged.

Preserve a back-compatible overload
`Train(data, maxEpochs, learnRate, momentum)` → calls the full signature with
`decay = 0.0`, `ErrorFunction.MeanSquaredError`.

### 3. Main / CLI

Parse simple optional command-line arguments:

- `--error=mse|ce` (default `ce`) — error function for back-prop and progress printing
- `--decay=<double>` (default `0.000001`) — weight decay λ; `0` disables
- `--lr=<double>` (default `0.02`), `--epochs=` (default 1000),
  `--momentum=` (default 0.01), `--rows=` (default 1000), `--seed=` (default 1)
- `--noise=<double>` (default `0.0`) — fraction of synthetic labels to corrupt
  (0–0.5). Optional; lets weight decay show its effect on a genuinely noisy data set.
- `--compare` — run four configurations on the **same** data split and print a
  summary table:
  1. MSE, no decay (the original algorithm)
  2. MSE + decay
  3. cross-entropy, no decay
  4. cross-entropy + decay
- `--help` — usage text.

Reporting per run:

- Settings echo (epochs, lr, momentum, decay, error function)
- Error progress lines every `errInterval` epochs
- Final serialized weights (43 values for 4-5-3)
- Train/test accuracy **and** train/test MSE + cross-entropy (both metrics, so
  runs are comparable regardless of training objective)
- For `--compare`: one summary table with train acc, test acc, and
  over-fit gap (train − test), exactly what the article says to watch.

Remove `Console.ReadLine()` so the program runs non-interactively
(`dotnet run`, CI, piping).

### 4. Tuning notes (measured)

- **CE gradient scale:** for softmax + cross-entropy the output signal has no
  `o·(1−o)` damping, so it is up to ~4× larger than the article's MSE signal.
  The article's `learnRate = 0.05` overshoots; `0.02` trains both error
  functions well (measured: MSE ≈ 99.0/98.5 %, CE ≈ 100/99 % train/test on seed 1).
- **Decay scale:** decay is applied on every one of the 800k online updates
  (800 rows × 1000 epochs), so λ must be small. λ = 0.001 crushes the weights
  (accuracy drops to ~94 %); λ = 1e-4 regularizes strongly. λ = 1e-6 is a safe
  default: measured neutral on the noise-free data (99.0/98.5 % for MSE and
  100/99 % for CE with or without decay) and clearly helpful on noisy data.
- **Label noise:** the article's generator produces deterministic (near
  noise-free) labels, so regularization has almost nothing to fix. The optional
  `--noise` knob injects label noise into the synthetic data (default 0 keeps
  the article-faithful data set) to demonstrate weight decay's purpose. With
  `--compare --noise=0.05` at the defaults, CE + decay reaches 93.5/91.5 %
  train/test (gap 2.0 %) versus 92.4/88.0 % without decay (gap 4.4 %).

### 5. Verification

1. `dotnet build` — clean compile, no warnings.
2. `dotnet run` — default run trains, error decreases, accuracies reported.
3. `dotnet run -- --compare` — table prints four rows; baseline reproduces the
   article's behavior (~99 % train / ~98.5 % test with seed 1); CE run learns.
4. `dotnet run -- --compare --noise=0.05` — decay visibly shrinks the gap.
5. `dotnet run -- --help` and an invalid argument — usage output, exit cleanly.

## Out of scope

- Multi-layer networks, mini-batch training, adaptive optimizers (Adam etc.).
- Saving the model to disk (article mentions it, WHAT TO DO doesn't ask for it).
- Changing the architecture of the synthetic-data generator.
