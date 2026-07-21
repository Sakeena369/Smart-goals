using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SmartGoals.CSharp;

public sealed class AssociateGoalInputRow
{
    public DateTime BUSINESS_DATE { get; init; }
    public string STORE_NUMBER { get; init; } = "";
    public double HOURS_SINCE_OPEN_CORRECTED { get; init; }
    public double GOALS_SCHEDULED_HOURS_CORRECTED { get; init; }
    public double STORE_MANAGER_GOAL { get; init; }
    public double? SALES_PER_GOAL_HOUR { get; init; }

    public Dictionary<string, object?> Features { get; init; } = new();
}

public sealed class AssociateGoalPredictionRow
{
    public DateTime BUSINESS_DATE { get; init; }
    public string STORE_NUMBER { get; init; } = "";
    public double HOURS_SINCE_OPEN_CORRECTED { get; init; }
    public double GOALS_SCHEDULED_HOURS_CORRECTED { get; init; }
    public double STORE_MANAGER_GOAL { get; init; }

    public double PREDICTED_SALES_PER_GOAL_HOUR { get; init; }
    public double ASSOCIATE_PCT_OF_PREDICTED { get; init; }
    public double ASSOCIATE_PCT_OF_HOURS { get; init; }
    public double ASSOCIATE_PCT_BLENDED { get; init; }
    public double SMART_GOAL { get; init; }

    public double? RESIDUAL { get; init; }
    public double? ABSOLUTE_ERROR { get; init; }
    public double? PERCENTAGE_ERROR { get; init; }
}

public static class SmartGoalsOnnxPredictor
{
    public static List<AssociateGoalPredictionRow> Predict(
        InferenceSession session,
        IReadOnlyList<AssociateGoalInputRow> data,
        double alpha = 0.5)
    {
        var n = data.Count;
        var modelInputs = BuildModelInputs(session, data, n);

        double[] predictions;
        using (var outputs = session.Run(modelInputs))
        {
            var firstOutput = outputs.First();
            predictions = ReadPredictions(firstOutput, n);
        }

        var result = new List<AssociateGoalPredictionRow>(n);
        var grouped = Enumerable.Range(0, n).GroupBy(i => (
            data[i].BUSINESS_DATE,
            data[i].STORE_NUMBER,
            data[i].HOURS_SINCE_OPEN_CORRECTED
        ));

        foreach (var group in grouped)
        {
            var sumPred = group.Sum(i => predictions[i]);
            var sumHours = group.Sum(i => data[i].GOALS_SCHEDULED_HOURS_CORRECTED);

            foreach (var i in group)
            {
                var predPct = predictions[i] / sumPred;
                var hoursPct = data[i].GOALS_SCHEDULED_HOURS_CORRECTED / sumHours;
                var blendedPct = alpha * predPct + (1.0 - alpha) * hoursPct;
                var smartGoal = data[i].STORE_MANAGER_GOAL * blendedPct;

                double? residual = null;
                double? absError = null;
                double? pctError = null;
                if (data[i].SALES_PER_GOAL_HOUR.HasValue)
                {
                    residual = data[i].SALES_PER_GOAL_HOUR.Value - predictions[i];
                    absError = Math.Abs(residual.Value);
                    pctError = (residual.Value / data[i].SALES_PER_GOAL_HOUR.Value) * 100.0;
                }

                result.Add(new AssociateGoalPredictionRow
                {
                    BUSINESS_DATE = data[i].BUSINESS_DATE,
                    STORE_NUMBER = data[i].STORE_NUMBER,
                    HOURS_SINCE_OPEN_CORRECTED = data[i].HOURS_SINCE_OPEN_CORRECTED,
                    GOALS_SCHEDULED_HOURS_CORRECTED = data[i].GOALS_SCHEDULED_HOURS_CORRECTED,
                    STORE_MANAGER_GOAL = data[i].STORE_MANAGER_GOAL,
                    PREDICTED_SALES_PER_GOAL_HOUR = predictions[i],
                    ASSOCIATE_PCT_OF_PREDICTED = predPct,
                    ASSOCIATE_PCT_OF_HOURS = hoursPct,
                    ASSOCIATE_PCT_BLENDED = blendedPct,
                    SMART_GOAL = smartGoal,
                    RESIDUAL = residual,
                    ABSOLUTE_ERROR = absError,
                    PERCENTAGE_ERROR = pctError,
                });
            }
        }

        return result;
    }

    private static List<NamedOnnxValue> BuildModelInputs(
        InferenceSession session,
        IReadOnlyList<AssociateGoalInputRow> data,
        int n)
    {
        var modelInputs = new List<NamedOnnxValue>();

        foreach (var (inputName, meta) in session.InputMetadata)
        {
            switch (meta.ElementType)
            {
                case TensorElementType.Float:
                    modelInputs.Add(NamedOnnxValue.CreateFromTensor(
                        inputName,
                        CreateTensor(data, n, inputName, Convert.ToSingle)));
                    break;
                case TensorElementType.Double:
                    modelInputs.Add(NamedOnnxValue.CreateFromTensor(
                        inputName,
                        CreateTensor(data, n, inputName, Convert.ToDouble)));
                    break;
                case TensorElementType.Int64:
                    modelInputs.Add(NamedOnnxValue.CreateFromTensor(
                        inputName,
                        CreateTensor(data, n, inputName, Convert.ToInt64)));
                    break;
                case TensorElementType.Int32:
                    modelInputs.Add(NamedOnnxValue.CreateFromTensor(
                        inputName,
                        CreateTensor(data, n, inputName, Convert.ToInt32)));
                    break;
                case TensorElementType.Bool:
                    modelInputs.Add(NamedOnnxValue.CreateFromTensor(
                        inputName,
                        CreateTensor(data, n, inputName, Convert.ToBoolean)));
                    break;
                case TensorElementType.String:
                    modelInputs.Add(NamedOnnxValue.CreateFromTensor(
                        inputName,
                        CreateTensor(data, n, inputName, v => Convert.ToString(v, CultureInfo.InvariantCulture) ?? "")));
                    break;
                default:
                    throw new NotSupportedException($"Unsupported ONNX input type for {inputName}: {meta.ElementType}");
            }
        }

        return modelInputs;
    }

    private static DenseTensor<T> CreateTensor<T>(
        IReadOnlyList<AssociateGoalInputRow> data,
        int n,
        string inputName,
        Func<object?, T> caster)
    {
        var tensor = new DenseTensor<T>(new[] { n, 1 });
        for (var i = 0; i < n; i++)
        {
            tensor[i, 0] = caster(data[i].Features[inputName]);
        }

        return tensor;
    }

    private static double[] ReadPredictions(DisposableNamedOnnxValue output, int n)
    {
        try
        {
            var values = output.AsTensor<float>().ToArray();
            return values.Take(n).Select(v => (double)v).ToArray();
        }
        catch
        {
            var values = output.AsTensor<double>().ToArray();
            return values.Take(n).ToArray();
        }
    }
}
