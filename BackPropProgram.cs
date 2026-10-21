using System;
using System.Collections.Generic;
using System.Globalization;
namespace CodingBackProp
{
  public enum ErrorFunction
  {
    MeanSquaredError, // original article: back-prop minimizing mean squared error
    CrossEntropy      // WHAT TO DO #1: back-prop minimizing cross-entropy error
  }

  class BackPropProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin neural network back-propagation demo");

      Options opt = new Options();
      if (!TryParseArgs(args, opt))
      {
        Console.WriteLine("\nBad command-line argument.\n");
        ShowUsage();
        return;
      }
      if (opt.Help)
      {
        ShowUsage();
        return;
      }

      int numInput = 4; // number features
      int numHidden = 5;
      int numOutput = 3; // number of classes for Y

      Console.WriteLine("\nGenerating " + opt.NumRows +
        " artificial data items with " + numInput + " features" +
        (opt.Noise > 0.0 ? " (" + (opt.Noise * 100.0).ToString("F1") +
          "% label noise)" : ""));
      double[][] allData = MakeAllData(numInput, numHidden, numOutput,
        opt.NumRows, opt.Seed, opt.Noise);
      Console.WriteLine("Done");

      //ShowMatrix(allData, allData.Length, 2, true);

      Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
      double[][] trainData;
      double[][] testData;
      SplitTrainTest(allData, 0.80, opt.Seed, out trainData, out testData);
      Console.WriteLine("Done\n");

      Console.WriteLine("Training data:");
      ShowMatrix(trainData, 4, 2, true);
      Console.WriteLine("Test data:");
      ShowMatrix(testData, 4, 2, true);

      if (opt.Compare)
      {
        RunCompare(numInput, numHidden, numOutput, trainData, testData, opt);
      }
      else
      {
        TrainAndReport("single run", opt.Error, opt.Decay, true,
          numInput, numHidden, numOutput, trainData, testData, opt);
      }

      Console.WriteLine("\nEnd back-propagation demo\n");
    } // Main

    private class Options // command-line settings
    {
      public int NumRows = 1000;
      public int Seed = 1;
      public int MaxEpochs = 1000;
      public double LearnRate = 0.02;
      public double Momentum = 0.01;
      public double Decay = 0.000001;
      public double Noise = 0.0;
      public ErrorFunction Error = ErrorFunction.CrossEntropy;
      public bool Compare = false;
      public bool Help = false;
    }

    private class RunResult // metrics for one training run
    {
      public string Label = "";
      public double TrainAcc;
      public double TestAcc;
      public double TrainMse;
      public double TestMse;
      public double TrainCe;
      public double TestCe;
    }

    static bool TryParseArgs(string[] args, Options opt)
    {
      foreach (string arg in args)
      {
        if (arg == "--compare") { opt.Compare = true; continue; }
        if (arg == "--help" || arg == "-h") { opt.Help = true; continue; }

        int eq = arg.IndexOf('=');
        if (!arg.StartsWith("--") || eq < 2) return false;
        string name = arg.Substring(2, eq - 2);
        string val = arg.Substring(eq + 1);

        switch (name)
        {
          case "error":
            switch (val.ToLowerInvariant())
            {
              case "mse": opt.Error = ErrorFunction.MeanSquaredError; break;
              case "ce": opt.Error = ErrorFunction.CrossEntropy; break;
              default: return false;
            }
            break;
          case "decay":
            if (!TryParseDouble(val, out opt.Decay) || opt.Decay < 0.0)
              return false;
            break;
          case "epochs":
            if (!int.TryParse(val, out opt.MaxEpochs) || opt.MaxEpochs < 1)
              return false;
            break;
          case "lr":
            if (!TryParseDouble(val, out opt.LearnRate) || opt.LearnRate <= 0.0)
              return false;
            break;
          case "momentum":
            if (!TryParseDouble(val, out opt.Momentum) || opt.Momentum < 0.0)
              return false;
            break;
          case "rows":
            if (!int.TryParse(val, out opt.NumRows) || opt.NumRows < 10)
              return false;
            break;
          case "seed":
            if (!int.TryParse(val, out opt.Seed)) return false;
            break;
          case "noise":
            if (!TryParseDouble(val, out opt.Noise) ||
              opt.Noise < 0.0 || opt.Noise > 0.5) return false;
            break;
          default:
            return false;
        }
      }
      return true;
    }

    static bool TryParseDouble(string s, out double d)
    {
      return double.TryParse(s, NumberStyles.Float,
        CultureInfo.InvariantCulture, out d);
    }

    static void ShowUsage()
    {
      Console.WriteLine("Usage: dotnet run -- [--options]");
      Console.WriteLine("  --error=mse|ce      training error function    (default: ce)");
      Console.WriteLine("  --decay=<double>    L2 weight decay, 0 = off   (default: 0.000001)");
      Console.WriteLine("  --epochs=<int>      max training epochs         (default: 1000)");
      Console.WriteLine("  --lr=<double>       learning rate               (default: 0.02)");
      Console.WriteLine("  --momentum=<double> momentum                    (default: 0.01)");
      Console.WriteLine("  --rows=<int>        synthetic data rows         (default: 1000)");
      Console.WriteLine("  --seed=<int>        random seed                 (default: 1)");
      Console.WriteLine("  --noise=<double>    fraction of labels to corrupt, 0-0.5 (default: 0)");
      Console.WriteLine("  --compare           MSE vs MSE+decay vs CE vs CE+decay summary");
      Console.WriteLine("  --help              show this help");
      Console.WriteLine();
      Console.WriteLine("WHAT TO DO modifications (instructions.md):");
      Console.WriteLine("  1. cross-entropy error instead of mean squared error");
      Console.WriteLine("  2. weight decay (regularization) to reduce over-fitting");
    }

    static RunResult TrainAndReport(string label, ErrorFunction errorFn,
      double decay, bool showWeights, int numInput, int numHidden,
      int numOutput, double[][] trainData, double[][] testData, Options opt)
    {
      Console.WriteLine("Creating a " + numInput + "-" + numHidden +
        "-" + numOutput + " neural network");
      NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

      Console.WriteLine("\nSetting maxEpochs = " + opt.MaxEpochs);
      Console.WriteLine("Setting learnRate = " + opt.LearnRate.ToString("F2"));
      Console.WriteLine("Setting momentum  = " + opt.Momentum.ToString("F2"));
      Console.WriteLine("Setting decay     = " + decay.ToString("0.######"));
      Console.WriteLine("Setting error     = " +
        (errorFn == ErrorFunction.CrossEntropy
          ? "cross-entropy" : "mean squared error"));

      Console.WriteLine("\nStarting training");
      double[] weights = nn.Train(trainData, opt.MaxEpochs, opt.LearnRate,
        opt.Momentum, decay, errorFn);
      Console.WriteLine("Done");

      if (showWeights)
      {
        Console.WriteLine("\nFinal neural network model weights and biases:\n");
        ShowVector(weights, 2, 10, true);
      }

      RunResult r = new RunResult();
      r.Label = label;
      r.TrainAcc = nn.Accuracy(trainData);
      r.TestAcc = nn.Accuracy(testData);
      r.TrainMse = nn.Error(trainData);
      r.TestMse = nn.Error(testData);
      r.TrainCe = nn.CrossEntropyError(trainData);
      r.TestCe = nn.CrossEntropyError(testData);

      Console.WriteLine("\nFinal accuracy on training data = " +
        r.TrainAcc.ToString("F4"));
      Console.WriteLine("Final accuracy on test data     = " +
        r.TestAcc.ToString("F4"));
      Console.WriteLine("Final MSE on train / test          = " +
        r.TrainMse.ToString("F6") + " / " + r.TestMse.ToString("F6"));
      Console.WriteLine("Final cross-entropy on train / test = " +
        r.TrainCe.ToString("F6") + " / " + r.TestCe.ToString("F6"));
      Console.WriteLine();
      return r;
    }

    static void RunCompare(int numInput, int numHidden, int numOutput,
      double[][] trainData, double[][] testData, Options opt)
    {
      Console.WriteLine("\n=== WHAT TO DO comparison " +
        "(same data, same split, same settings) ===\n");

      List<RunResult> results = new List<RunResult>();
      double decay = opt.Decay;
      string decayLabel = decay.ToString("0.######");
      results.Add(TrainAndReport("MSE, no decay",
        ErrorFunction.MeanSquaredError, 0.0, false,
        numInput, numHidden, numOutput, trainData, testData, opt));
      results.Add(TrainAndReport("MSE + decay " + decayLabel,
        ErrorFunction.MeanSquaredError, decay, false,
        numInput, numHidden, numOutput, trainData, testData, opt));
      results.Add(TrainAndReport("cross-entropy, no decay",
        ErrorFunction.CrossEntropy, 0.0, false,
        numInput, numHidden, numOutput, trainData, testData, opt));
      results.Add(TrainAndReport("cross-entropy + decay " + decayLabel,
        ErrorFunction.CrossEntropy, decay, false,
        numInput, numHidden, numOutput, trainData, testData, opt));

      Console.WriteLine("Summary (over-fit gap = train acc - test acc):");
      Console.WriteLine("  configuration                   train acc   test acc       gap");
      foreach (RunResult r in results)
      {
        double gap = r.TrainAcc - r.TestAcc;
        Console.WriteLine("  " + r.Label.PadRight(31) +
          r.TrainAcc.ToString("F4").PadLeft(10) +
          r.TestAcc.ToString("F4").PadLeft(11) +
          gap.ToString("+0.0000;-0.0000").PadLeft(9));
      }
      Console.WriteLine();
    }

    public static void ShowMatrix(double[][] matrix, int numRows,
      int decimals, bool indices)
    {
      int len = matrix.Length.ToString().Length;
      for (int i = 0; i < numRows; ++i)
      {
        if (indices == true)
          Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < matrix[i].Length; ++j)
        {
          double v = matrix[i][j];
          if (v >= 0.0)
            Console.Write(" "); // '+'
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
        Console.WriteLine("");
      }

      if (numRows < matrix.Length)
      {
        Console.WriteLine(". . .");
        int lastRow = matrix.Length - 1;
        if (indices == true)
          Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
        for (int j = 0; j < matrix[lastRow].Length; ++j)
        {
          double v = matrix[lastRow][j];
          if (v >= 0.0)
            Console.Write(" "); // '+'
          Console.Write(v.ToString("F" + decimals) + "  ");
        }
      }
      Console.WriteLine("\n");
    }

    public static void ShowVector(double[] vector, int decimals,
      int lineLen, bool newLine)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i > 0 && i % lineLen == 0) Console.WriteLine("");
        if (vector[i] >= 0) Console.Write(" ");
        Console.Write(vector[i].ToString("F" + decimals) + " ");
      }
      if (newLine == true)
        Console.WriteLine("");
    }

    static double[][] MakeAllData(int numInput, int numHidden,
      int numOutput, int numRows, int seed, double labelNoise = 0.0)
    {
      Random rnd = new Random(seed);
      Random noiseRnd = new Random(seed + 1); // separate stream, used only if noisy
      int numWeights = (numInput * numHidden) + numHidden +
        (numHidden * numOutput) + numOutput;
      double[] weights = new double[numWeights]; // actually weights & biases
      for (int i = 0; i < numWeights; ++i)
        weights[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to 10.0]

      Console.WriteLine("Generating weights and biases:");
      ShowVector(weights, 2, 10, true);

      double[][] result = new double[numRows][]; // allocate return-result
      for (int i = 0; i < numRows; ++i)
        result[i] = new double[numInput + numOutput]; // 1-of-N in last column

      NeuralNetwork gnn =
        new NeuralNetwork(numInput, numHidden, numOutput); // generating NN
      gnn.SetWeights(weights);

      for (int r = 0; r < numRows; ++r) // for each row
      {
        // generate random inputs
        double[] inputs = new double[numInput];
        for (int i = 0; i < numInput; ++i)
          inputs[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to -10.0]

        // compute outputs
        double[] outputs = gnn.ComputeOutputs(inputs);

        // translate outputs to 1-of-N
        double[] oneOfN = new double[numOutput]; // all 0.0

        int maxIndex = 0;
        double maxValue = outputs[0];
        for (int i = 0; i < numOutput; ++i)
        {
          if (outputs[i] > maxValue)
          {
            maxIndex = i;
            maxValue = outputs[i];
          }
        }

        // optionally corrupt a fraction of the labels (inject noise)
        // so that over-fitting, and therefore weight decay, matters
        if (labelNoise > 0.0 && noiseRnd.NextDouble() < labelNoise)
        {
          int noisyIndex = noiseRnd.Next(numOutput - 1);
          if (noisyIndex >= maxIndex) ++noisyIndex; // a different class
          maxIndex = noisyIndex;
        }

        oneOfN[maxIndex] = 1.0;

        // place inputs and 1-of-N output values into curr row
        int c = 0; // column into result[][]
        for (int i = 0; i < numInput; ++i) // inputs
          result[r][c++] = inputs[i];
        for (int i = 0; i < numOutput; ++i) // outputs
          result[r][c++] = oneOfN[i];
      } // each row
      return result;
    } // MakeAllData

    static void SplitTrainTest(double[][] allData, double trainPct,
      int seed, out double[][] trainData, out double[][] testData)
    {
      Random rnd = new Random(seed);
      int totRows = allData.Length;
      int numTrainRows = (int)(totRows * trainPct); // usually 0.80
      int numTestRows = totRows - numTrainRows;
      trainData = new double[numTrainRows][];
      testData = new double[numTestRows][];

      double[][] copy = new double[allData.Length][]; // ref copy of data
      for (int i = 0; i < copy.Length; ++i)
        copy[i] = allData[i];

      for (int i = 0; i < copy.Length; ++i) // scramble order
      {
        int r = rnd.Next(i, copy.Length); // use Fisher-Yates
        double[] tmp = copy[r];
        copy[r] = copy[i];
        copy[i] = tmp;
      }
      for (int i = 0; i < numTrainRows; ++i)
        trainData[i] = copy[i];

      for (int i = 0; i < numTestRows; ++i)
        testData[i] = copy[i + numTrainRows];
    } // SplitTrainTest

  } // Program

  public class NeuralNetwork
  {
    private int numInput; // number input nodes
    private int numHidden;
    private int numOutput;

    private double[] inputs;
    private double[][] ihWeights; // input-hidden
    private double[] hBiases;
    private double[] hOutputs;

    private double[][] hoWeights; // hidden-output
    private double[] oBiases;
    private double[] outputs;

    private Random rnd;

    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      this.inputs = new double[numInput];

      this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
      this.hBiases = new double[numHidden];
      this.hOutputs = new double[numHidden];

      this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
      this.oBiases = new double[numOutput];
      this.outputs = new double[numOutput];

      this.rnd = new Random(0);
      this.InitializeWeights(); // all weights and biases
    } // ctor

    private static double[][] MakeMatrix(int rows,
      int cols, double v) // helper for ctor, Train
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < result.Length; ++r)
        result[r] = new double[cols];
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = v;
      return result;
    }

    //private static double[][] MakeMatrixRandom(int rows,
    //  int cols, int seed) // helper for ctor, Train
    //{
    //  Random rnd = new Random(seed);
    //  double hi = 0.01;
    //  double lo = -0.01;
    //  double[][] result = new double[rows][];
    //  for (int r = 0; r < result.Length; ++r)
    //    result[r] = new double[cols];
    //  for (int i = 0; i < rows; ++i)
    //    for (int j = 0; j < cols; ++j)
    //      result[i][j] = (hi - lo) * rnd.NextDouble() + lo;
    //  return result;
    //}

    private void InitializeWeights() // helper for ctor
    {
      // initialize weights and biases to small random values
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] initialWeights = new double[numWeights];
      for (int i = 0; i < initialWeights.Length; ++i)
        initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
      this.SetWeights(initialWeights);
    }

    public void SetWeights(double[] weights)
    {
      // copy serialized weights and biases in weights[] array
      // to i-h weights, i-h biases, h-o weights, h-o biases
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      if (weights.Length != numWeights)
        throw new Exception("Bad weights array in SetWeights");

      int k = 0; // points into weights param

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];
      for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
    }

    public double[] GetWeights()
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] result = new double[numWeights];
      int k = 0;
      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < hBiases.Length; ++i)
        result[k++] = hBiases[i];
      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < oBiases.Length; ++i)
        result[k++] = oBiases[i];
      return result;
    }

    public double[] ComputeOutputs(double[] xValues)
    {
      double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
      double[] oSums = new double[numOutput]; // output nodes sums

      for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
        this.inputs[i] = xValues[i];
      // note: no need to copy x-values unless you implement a ToString.
      // more efficient is to simply use the xValues[] directly.

      for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
        for (int i = 0; i < numInput; ++i)
          hSums[j] += this.inputs[i] * this.ihWeights[i][j]; // note +=

      for (int i = 0; i < numHidden; ++i)  // add biases to hidden sums
        hSums[i] += this.hBiases[i];

      for (int i = 0; i < numHidden; ++i)   // apply activation
        this.hOutputs[i] = HyperTan(hSums[i]); // hard-coded

      for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += hOutputs[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)  // add biases to output sums
        oSums[i] += oBiases[i];

      double[] softOut = Softmax(oSums); // all outputs at once for efficiency
      Array.Copy(softOut, outputs, softOut.Length);

      double[] retResult = new double[numOutput]; // could define a GetOutputs 
      Array.Copy(this.outputs, retResult, retResult.Length);
      return retResult;
    }

    private static double HyperTan(double x)
    {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
    }

    private static double[] Softmax(double[] oSums)
    {
      // does all output nodes at once so scale
      // doesn't have to be re-computed each time

      double sum = 0.0;
      for (int i = 0; i < oSums.Length; ++i)
        sum += Math.Exp(oSums[i]);

      double[] result = new double[oSums.Length];
      for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i]) / sum;

      return result; // now scaled so that xi sum to 1.0
    }

    public double[] Train(double[][] trainData, int maxEpochs,
      double learnRate, double momentum)
    {
      // back-compatible overload: original article behavior
      // (mean squared error, no weight decay)
      return Train(trainData, maxEpochs, learnRate, momentum, 0.0,
        ErrorFunction.MeanSquaredError);
    }

    public double[] Train(double[][] trainData, int maxEpochs,
      double learnRate, double momentum, double decay,
      ErrorFunction errorFunction)
    {
      // train using back-prop
      // back-prop specific arrays
      double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weight gradients
      double[] obGrads = new double[numOutput];                   // output bias gradients

      double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);  // input-to-hidden weight gradients
      double[] hbGrads = new double[numHidden];                   // hidden bias gradients

      double[] oSignals = new double[numOutput];                  // local gradient output signals - gradients w/o associated input terms
      double[] hSignals = new double[numHidden];                  // local gradient hidden node signals

      // back-prop momentum specific arrays 
      double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
      double[] hPrevBiasesDelta = new double[numHidden];
      double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
      double[] oPrevBiasesDelta = new double[numOutput];

      int epoch = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // target values
      double derivative = 0.0;
      double errorSignal = 0.0;

      int[] sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      int errInterval = Math.Max(1, maxEpochs / 10); // interval to check error
      while (epoch < maxEpochs)
      {
        ++epoch;

        if (epoch % errInterval == 0 && epoch < maxEpochs)
        {
          double trainErr = (errorFunction == ErrorFunction.CrossEntropy)
            ? CrossEntropyError(trainData) : Error(trainData);
          Console.WriteLine("epoch = " + epoch + "  error = " +
            trainErr.ToString("F6"));
          //Console.ReadLine();
        }

        Shuffle(sequence); // visit each training data in random order
        for (int ii = 0; ii < trainData.Length; ++ii)
        {
          int idx = sequence[ii];
          Array.Copy(trainData[idx], xValues, numInput);
          Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
          ComputeOutputs(xValues); // copy xValues in, compute outputs 

          // indices: i = inputs, j = hiddens, k = outputs

          // 1. compute output node signals (assumes softmax)
          for (int k = 0; k < numOutput; ++k)
          {
            errorSignal = tValues[k] - outputs[k];  // Wikipedia uses (o-t)
            if (errorFunction == ErrorFunction.CrossEntropy)
            {
              // WHAT TO DO #1: cross-entropy + softmax. The output
              // derivative term cancels exactly, so the signal is
              // simply the error: oSignal = t - o
              oSignals[k] = errorSignal;
            }
            else
            {
              derivative = (1 - outputs[k]) * outputs[k]; // for softmax + MSE
              oSignals[k] = errorSignal * derivative;
            }
          }
  
          // 2. compute hidden-to-output weight gradients using output signals
          for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
               hoGrads[j][k] = oSignals[k] * hOutputs[j];
 
          // 2b. compute output bias gradients using output signals
          for (int k = 0; k < numOutput; ++k)
            obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

          // 3. compute hidden node signals
          for (int j = 0; j < numHidden; ++j)
          {
            derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // for tanh
            double sum = 0.0; // need sums of output signals times hidden-to-output weights
            for (int k = 0; k < numOutput; ++k) {
               sum += oSignals[k] * hoWeights[j][k]; // represents error signal
            }
            hSignals[j] = derivative * sum; 
          }

          // 4. compute input-hidden weight gradients
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihGrads[i][j] = hSignals[j] * inputs[i];

          // 4b. compute hidden node bias gradients
          for (int j = 0; j < numHidden; ++j)
            hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

          // == update weights and biases

          // update input-to-hidden weights
          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              // WHAT TO DO #2: weight decay. gradient of an extra
              // (decay/2)*w^2 penalty term, scaled by learnRate
              double delta = ihGrads[i][j] * learnRate
                - ihWeights[i][j] * learnRate * decay;
              ihWeights[i][j] += delta; // would be -= if (o-t)
              ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
              ihPrevWeightsDelta[i][j] = delta; // save for next time
            }
          }

          // update hidden biases
          for (int j = 0; j < numHidden; ++j)
          {
            double delta = hbGrads[j] * learnRate;
            hBiases[j] += delta;
            hBiases[j] += hPrevBiasesDelta[j] * momentum;
            hPrevBiasesDelta[j] = delta;
          }

          // update hidden-to-output weights
          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              // WHAT TO DO #2: weight decay (biases are not decayed)
              double delta = hoGrads[j][k] * learnRate
                - hoWeights[j][k] * learnRate * decay;
              hoWeights[j][k] += delta;
              hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
              hoPrevWeightsDelta[j][k] = delta;
            }
          }

          // update output node biases
          for (int k = 0; k < numOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
            oBiases[k] += oPrevBiasesDelta[k] * momentum;
            oPrevBiasesDelta[k] = delta;
          }

        } // each training item

      } // while
      double[] bestWts = GetWeights();
      return bestWts;
    } // Train

    private void Shuffle(int[] sequence) // instance method
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = this.rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    } // Shuffle

    public double Error(double[][] trainData)
    {
      // average squared error per training item
      double sumSquaredError = 0.0;
      double[] xValues = new double[numInput]; // first numInput values in trainData
      double[] tValues = new double[numOutput]; // last numOutput values

      // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
      for (int i = 0; i < trainData.Length; ++i)
      {
        Array.Copy(trainData[i], xValues, numInput);
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
        double[] yValues = this.ComputeOutputs(xValues); // outputs using current weights
        for (int j = 0; j < numOutput; ++j)
        {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }
      return sumSquaredError / trainData.Length;
    } // MeanSquaredError

    public double CrossEntropyError(double[][] data)
    {
      // WHAT TO DO #1: mean cross-entropy error per data item
      // CE = -sum(t_k * ln(y_k)); with 1-of-N labels only the
      // target class term is non-zero, so the sum simplifies
      double sumLog = 0.0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];

      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, numInput);
        Array.Copy(data[i], numInput, tValues, 0, numOutput);
        double[] yValues = this.ComputeOutputs(xValues);
        for (int j = 0; j < numOutput; ++j)
        {
          if (tValues[j] > 0.0)
          {
            double p = Math.Max(yValues[j], 1.0e-12); // guard ln(0)
            sumLog += tValues[j] * Math.Log(p);
          }
        }
      }
      return -sumLog / data.Length;
    } // CrossEntropyError

    public double Accuracy(double[][] testData)
    {
      // percentage correct using winner-takes all
      int numCorrect = 0;
      int numWrong = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // targets
      double[] yValues; // computed Y

      for (int i = 0; i < testData.Length; ++i)
      {
        Array.Copy(testData[i], xValues, numInput); // get x-values
        Array.Copy(testData[i], numInput, tValues, 0, numOutput); // get t-values
        yValues = this.ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
        int tMaxIndex = MaxIndex(tValues);

        if (maxIndex == tMaxIndex)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

    private static int MaxIndex(double[] vector) // helper for Accuracy()
    {
      // index of largest value
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 0; i < vector.Length; ++i)
      {
        if (vector[i] > biggestVal)
        {
          biggestVal = vector[i];
          bigIndex = i;
        }
      }
      return bigIndex;
    }


  } // NeuralNetwork



} // ns
