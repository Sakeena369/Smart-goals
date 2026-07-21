# C# ONNX Predictor

This folder contains a C# implementation that mirrors the Python inference flow in `src/smart_goals/predict.py`:

1. Score associate rows with a loaded ONNX model.
2. Compute `ASSOCIATE_PCT_OF_PREDICTED` by hour/store/date group.
3. Compute `ASSOCIATE_PCT_OF_HOURS` by the same group.
4. Blend with `alpha`:
   `ASSOCIATE_PCT_BLENDED = alpha * ASSOCIATE_PCT_OF_PREDICTED + (1 - alpha) * ASSOCIATE_PCT_OF_HOURS`
5. Compute `SMART_GOAL = STORE_MANAGER_GOAL * ASSOCIATE_PCT_BLENDED`.

## NuGet package

- `Microsoft.ML.OnnxRuntime`

## Usage

```csharp
using Microsoft.ML.OnnxRuntime;
using SmartGoals.CSharp;

using var session = new InferenceSession("models/smart_goals.onnx");

var rows = new List<AssociateGoalInputRow>
{
    new()
    {
        BUSINESS_DATE = new DateTime(2026, 1, 10),
        STORE_NUMBER = "4002",
        HOURS_SINCE_OPEN_CORRECTED = 1.0,
        GOALS_SCHEDULED_HOURS_CORRECTED = 8.0,
        STORE_MANAGER_GOAL = 3500.0,
        SALES_PER_GOAL_HOUR = null,
        Features = new Dictionary<string, object?>
        {
            // Must include every ONNX input by exact ONNX input name.
        }
    }
};

var scored = SmartGoalsOnnxPredictor.Predict(session, rows, alpha: 0.5);
```
